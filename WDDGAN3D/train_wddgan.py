import argparse
import os
import shutil

import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from prettytable import PrettyTable
from datasets_prep.dataset import create_dataset
from diffusion import sample_from_model, sample_posterior, \
    q_sample_pairs, get_time_schedule, \
    Posterior_Coefficients, Diffusion_Coefficients
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D, DWT_3D, IDWT_3D
from pytorch_wavelets import DWTForward, DWTInverse
from torch.multiprocessing import Process
from utils import init_processes, copy_source, broadcast_params

from torch.utils.tensorboard import SummaryWriter

from dataset import CreateDatasetSynthesis

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def grad_penalty_call(args, D_real, x_t):
    grad_real = torch.autograd.grad(
        outputs=D_real.sum(), inputs=x_t, create_graph=True
    )[0]
    grad_penalty = (
        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
    ).mean()

    grad_penalty = args.r1_gamma / 2 * grad_penalty
    grad_penalty.backward()
    print('grad_penalty: ', grad_penalty)


# %%
def train(rank, gpu, args):
    from EMA import EMA
    from score_sde.models.discriminator import Discriminator_large, Discriminator_small
    from score_sde.models.ncsnpp_generator_adagn import NCSNpp, WaveletNCSNpp


    timestr = time.strftime("%d_%m_%Y-%H_%M_%S")
    writer = SummaryWriter(log_dir=args.log_dir + timestr)

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))

    batch_size = args.batch_size

    nz = args.nz  # latent dimension

    if args.dataset=='brats':
        dataset = CreateDatasetSynthesis(folder1=args.datadir)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                        num_replicas=args.world_size,
                                                                        rank=rank)

        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  pin_memory=True,
                                                  sampler=train_sampler,
                                                  drop_last=True)

    else:
        dataset = create_dataset(args)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                        num_replicas=args.world_size,
                                                                        rank=rank)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True,
                                                  sampler=train_sampler,
                                                  drop_last=True)
    args.ori_image_size = args.image_size
    args.image_size = args.current_resolution
    G_NET_ZOO = {"normal": NCSNpp, "wavelet": WaveletNCSNpp}
    gen_net = G_NET_ZOO[args.net_type]
    disc_net = [Discriminator_small, Discriminator_large]
    netG = gen_net(args).to(device)

    if args.dataset in ['cifar10', 'stl10']:
        netD = disc_net[0](nc=2 * 1, ngf=args.ngf,
                           t_emb_dim=args.t_emb_dim,
                           act=nn.LeakyReLU(0.2), num_layers=args.num_disc_layers).to(device)
    else:
        netD = disc_net[0](nc=2 * 1, ngf=args.ngf,
                           t_emb_dim=args.t_emb_dim,
                           act=nn.LeakyReLU(0.2), num_layers=args.num_disc_layers).to(device)


    broadcast_params(netG.parameters())
    broadcast_params(netD.parameters())

    optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters(
    )), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(filter(lambda p: p.requires_grad, netG.parameters(
    )), lr=args.lr_g, betas=(args.beta1, args.beta2))

    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizerG, args.num_epoch, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizerD, args.num_epoch, eta_min=1e-5)

    # ddp
    netG = nn.parallel.DistributedDataParallel(
        netG, device_ids=[gpu], find_unused_parameters=False)
    netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu])

    # Wavelet Pooling
    if not args.use_pytorch_wavelet:
        dwt = DWT_3D("haar")
        iwt = IDWT_3D("haar")
    else:
        dwt = DWTForward(J=1, mode='zero', wave='haar').cuda()
        iwt = DWTInverse(mode='zero', wave='haar').cuda()

    num_levels = int(np.log2(args.ori_image_size // args.current_resolution))

    exp = args.exp
    name_dataset = args.dataset + '_' + timestr
    parent_dir = "./saved_info/wdd_gan/{}".format(name_dataset)

    exp_path = os.path.join(parent_dir, exp)

    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('score_sde/models',
                            os.path.join(exp_path, 'score_sde/models'))

    logging.basicConfig(
        filename=os.path.join(exp_path, 'arguments_used.log'),
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    logging.info('logging ')
    logging.info(timestr)
    logging.info(f'args used: {args}')

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)
    
    logging.info(f'coeff used: {coeff.__dict__}')
    logging.info(f'pos_coeff used: {pos_coeff.__dict__}')
    logging.info(f'T schedule: {T}')

    if args.resume or os.path.exists(os.path.join(exp_path, 'content.pth')):
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        # load G
        netG.load_state_dict(checkpoint['netG_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])

        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    for epoch in range(init_epoch, args.num_epoch + 1):
        train_sampler.set_epoch(epoch)

        for iteration, (x, y, mk) in enumerate(data_loader):

            for p in netD.parameters():
                p.requires_grad = True
            netD.zero_grad()

            for p in netG.parameters():
                p.requires_grad = False

            # sample from p(x_0)
            x0 = x.to(device, non_blocking=True)

            if not args.use_pytorch_wavelet:
                for i in range(num_levels):
                    LLL0, LLH0, LHL0, LHH0, HLL0, HLH0, HHL0, HHH0 = dwt(x0[:, 0, :, :, :].unsqueeze(1))
                    LLL1, LLH1, LHL1, LHH1, HLL, HLH1, HHL1, HHH1 = dwt(x0[:, 1, :, :, :].unsqueeze(1))
                    LLL2, LLH2, LHL2, LHH2, HLL2, HLH2, HHL2, HHH2= dwt(x0[:, 2, :, :, :].unsqueeze(1))
            else:
                LLL, xh = dwt(x0)  # [b, 3, h, w], [b, 3, 3, h, w]
                LLH, LHL, LHH, HLL, HLH, HHL, HHH = torch.unbind(xh[0], dim=2)

            real_data = torch.cat([LLL0, LLH0, LHL0, LHH0, HLL0, HLH0, HHL0, HHH0, LLL1, LLH1, LHL1, LHH1, HLL, HLH1, HHL1, HHH1, LLL2, LLH2, LHL2, LHH2, HLL2, HLH2, HHL2, HHH2], dim=1)  # [b, 12, h, w]

            # normalize real_data
            real_data = real_data / np.sqrt(8.0)  # [-1, 1]
            real_data = real_data.clamp(-1, 1)

            assert -1 <= real_data.min() < 0
            assert 0 < real_data.max() <= 1

            # sample t
            t = torch.randint(0, args.num_timesteps,
                              (real_data.size(0),), device=device)

            x_t, x_tp1 = q_sample_pairs(coeff, real_data[:, 16:24, :, :, :], t)
            x_t = torch.cat((real_data[:, :16, :, :, :], x_t), dim=1)
            x_tp1 = torch.cat((real_data[:, :16, :, :, :], x_tp1), dim=1)

            x_t.requires_grad = True

            # train with real

            x_t_IWT = iwt(
                    x_t[:, 16:17], x_t[:, 17:18], x_t[:, 18:19], x_t[:, 19:20],
                    x_t[:, 20:21], x_t[:, 21:22], x_t[:, 22:23], x_t[:, 23:24])

            x_tp1_IWT = iwt(
                    x_tp1[:, 16:17], x_tp1[:, 17:18], x_tp1[:, 18:19], x_tp1[:, 19:20],
                    x_tp1[:, 20:21], x_tp1[:, 21:22], x_tp1[:, 22:23], x_tp1[:, 23:24]).detach()
                    
            x_t_IWT = x_t_IWT[..., 32:-32,32:-32,32:-32]
            x_tp1_IWT = x_tp1_IWT[..., 32:-32,32:-32,32:-32]

            D_real = netD(x_t_IWT, t, x_tp1_IWT)
            errD_real = F.softplus(-D_real).mean()
            errD_real.backward(retain_graph=True)

            if args.lazy_reg is None:
                grad_penalty_call(args, D_real, x_t)
            else:
                if global_step % args.lazy_reg == 0:
                    grad_penalty_call(args, D_real, x_t)

            # train with fake

            latent_z = torch.randn(batch_size, nz, device=device)
            x_0_predict = netG(x_tp1.detach(), t, latent_z)

            x_tp1 = x_tp1[:, 16:, :, :, :]
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            x_pos_sample_IWT = iwt(
                    x_pos_sample[:, :1], x_pos_sample[:, 1:2], x_pos_sample[:, 2:3], x_pos_sample[:, 3:4], x_pos_sample[:, 4:5], x_pos_sample[:, 5:6], x_pos_sample[:, 6:7],
                    x_pos_sample[:, 7:8])

            x_tp1_IWT = iwt(
                    x_tp1[:, :1], x_tp1[:, 1:2], x_tp1[:, 2:3], x_tp1[:, 3:4], x_tp1[:, 4:5], x_tp1[:, 5:6], x_tp1[:, 6:7],
                    x_tp1[:, 7:8]).detach()
                    
            x_pos_sample_IWT = x_pos_sample_IWT[..., 32:-32,32:-32,32:-32]
            x_tp1_IWT = x_tp1_IWT[..., 32:-32,32:-32,32:-32]

            output = netD(x_pos_sample_IWT, t, x_tp1_IWT) #netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            errD_fake = F.softplus(output).mean()

            errD_fake.backward()

            errD = errD_real + errD_fake

            # Update D
            optimizerD.step()

            if not args.no_lr_decay:
                schedulerD.step()

            # Update G multiple times
            for _ in range(args.num_generator_updates):
                for p in netD.parameters():
                    p.requires_grad = False

                for p in netG.parameters():
                    p.requires_grad = True
                netG.zero_grad()

                t = torch.randint(0, args.num_timesteps,
                                  (real_data.size(0),), device=device)
                x_t, x_tp1 = q_sample_pairs(coeff, real_data[:,16:24,:,:,:], t)
                x_t = torch.cat((real_data[:, :16, :, :, :], x_t), dim=1)
                x_tp1 = torch.cat((real_data[:, :16, :, :, :], x_tp1), dim=1)

                latent_z = torch.randn(batch_size, nz, device=device)
                latent_z = latent_z.clamp(-1, 1)
                x_0_predict = netG(x_tp1.detach(), t, latent_z)
                x0p = x_0_predict.clone()
                x_tp1 = x_tp1[:, 16:, :, :, :]
                x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
                xpos = x_pos_sample.clone()

                t_pos_sam = t

                x_pos_sample_IWT = iwt(
                    x_pos_sample[:, :1], x_pos_sample[:, 1:2], x_pos_sample[:, 2:3], x_pos_sample[:, 3:4],
                    x_pos_sample[:, 4:5], x_pos_sample[:, 5:6], x_pos_sample[:, 6:7],
                    x_pos_sample[:, 7:8])

                x_tp1_IWT = iwt(
                        x_tp1[:, :1], x_tp1[:, 1:2], x_tp1[:, 2:3], x_tp1[:, 3:4], x_tp1[:, 4:5], x_tp1[:, 5:6], x_tp1[:, 6:7],
                        x_tp1[:, 7:8]).detach()

                x_pos_sample_IWT = x_pos_sample_IWT[..., 32:-32,32:-32,32:-32]
                x_tp1_IWT = x_tp1_IWT[..., 32:-32,32:-32,32:-32]

                # Calculate generator loss
                output = netD(x_pos_sample_IWT, t, x_tp1_IWT)
                errG = F.softplus(-output).mean()

                # reconstruction losses

                x0p_rec = x_0_predict.clone()
                x0p_rec *= np.sqrt(8.0)

                x_0_predict_IWT = iwt(
                    x0p_rec[:, :1], x0p_rec[:, 1:2], x0p_rec[:, 2:3], x0p_rec[:, 3:4], x0p_rec[:, 4:5], x0p_rec[:, 5:6],
                    x0p_rec[:, 6:7],
                    x0p_rec[:, 7:8])

                mask_original = x0[:, 1, :, :, :].unsqueeze(1).type(torch.int)
                GT_original = x0[:, 2, :, :, :].unsqueeze(1)

                x_0_predict_IWT_masked = x_0_predict_IWT.clone()
                x_0_predict_IWT_masked_mask_only = x_0_predict_IWT_masked[mask_original == 1]

                GT_original_masked = GT_original.clone()
                GT_original_masked_mask_only = GT_original_masked[mask_original == 1]

                rec_loss_IWT_masked = F.l1_loss(x_0_predict_IWT_masked_mask_only, GT_original_masked_mask_only)
                rec_loss_IWT_full = F.l1_loss(x_0_predict_IWT, GT_original)

                errG_total = errG + rec_loss_IWT_full + rec_loss_IWT_masked
                errG_total.backward()

                optimizerG.step()

                if not args.no_lr_decay:
                    schedulerG.step()

            global_step += 1
            if iteration % 1 == 0:
                if rank == 0:
                    print('epoch {} iteration{}, G Loss: {}, D Loss: {}'.format(
                        epoch, iteration, errG_total.item(), errD.item()))

            writer.add_scalar('Loss/trainG_total', errG_total.item(), global_step)
            writer.add_scalar('Loss/trainG_adv', errG.item(), global_step)
            writer.add_scalar('Loss/trainG_rec_full', rec_loss_IWT_full.item(), global_step)
            writer.add_scalar('Loss/trainG_rec_mask', rec_loss_IWT_masked.item(), global_step)
            writer.add_scalar('Loss/trainD', errD.item(), global_step)

        #if rank == 0:

            if (global_step - 1) % 50 == 0:

                idx_show_0 = 64
                idx_show_1 = 64
                idx_show_2 = 64

                x_t_1 = torch.randn_like(real_data)
                fake_sample = sample_from_model(
                    pos_coeff, netG, args.num_timesteps, x_t_1, T, real_data, args)

                fake_sample *= np.sqrt(8.0)

                if not args.use_pytorch_wavelet:

                    fake_sample = iwt(
                        fake_sample[:, 0:1], fake_sample[:, 1:2], fake_sample[:, 2:3], fake_sample[:, 3:4], fake_sample[:, 4:5], fake_sample[:, 5:6], fake_sample[:, 6:7], fake_sample[:, 7:8])

                else:
                    fake_sample = iwt((fake_sample[:, :3], [torch.stack(
                        (fake_sample[:, 3:6], fake_sample[:, 6:9], fake_sample[:, 9:12]), dim=2)]))

                fake_sample = (torch.clamp(fake_sample, -1, 1) + 1) / 2  # 0-1
                real_data_0 = (torch.clamp(x0[:,0,:,:,:].unsqueeze(1), -1, 1) + 1) / 2  # 0-1
                real_data_1 = (torch.clamp(x0[:,1,:,:,:].unsqueeze(1), -1, 1) + 1) / 2  # 0-1
                real_data_2 = (torch.clamp(x0[:,2,:,:,:].unsqueeze(1), -1, 1) + 1) / 2  # 0-1

                writer.add_image('Images Real Voided /real_data_voided_0_0', real_data_0[0, :, idx_show_0, :, :], global_step-1)
                writer.add_image('Images Real Mask /real_data_mask_0_0', real_data_1[0, :, idx_show_0, :, :], global_step-1)
                writer.add_image('Images Real GT /real_data_t1n_0_0', real_data_2[0, :, idx_show_0, :, :], global_step-1)
                writer.add_image('Images Real Voided /real_data_voided_0_1', real_data_0[0, :, :, idx_show_1, :], global_step-1)
                writer.add_image('Images Real Mask /real_data_mask_0_1', real_data_1[0, :, :, idx_show_1, :], global_step-1)
                writer.add_image('Images Real GT /real_data_t1n_0_1', real_data_2[0, :, :, idx_show_1, :], global_step-1)
                writer.add_image('Images Real Voided /real_data_voided_0_2', real_data_0[0, :, :, :, idx_show_2], global_step-1)
                writer.add_image('Images Real Mask /real_data_mask_0_2', real_data_1[0, :, :, :, idx_show_2], global_step-1)
                writer.add_image('Images Real GT /real_data_t1n_0_2', real_data_2[0, :, :, :, idx_show_2], global_step-1)

                writer.add_image('Images Fake Sample /fake_sample_0_0', fake_sample[0, :, idx_show_0, :, :], global_step-1)
                writer.add_image('Images Fake Sample /fake_sample_0_1', fake_sample[0, :, :, idx_show_1, :], global_step-1)
                writer.add_image('Images Fake Sample /fake_sample_0_2', fake_sample[0, :, :, :, idx_show_2], global_step-1)

        if args.save_content:
            if epoch % args.save_content_every == 0:
                print('Saving content.')
                content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                           'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                           'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                           'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}
                torch.save(content, os.path.join(exp_path, 'content.pth'))

        if epoch % args.save_ckpt_every == 0:
            if args.use_ema:
                optimizerG.swap_parameters_with_ema(
                    store_params_in_ema=True)

            torch.save(netG.state_dict(), os.path.join(
                exp_path, 'netG_{}.pth'.format(epoch)))
            if args.use_ema:
                optimizerG.swap_parameters_with_ema(
                    store_params_in_ema=True)


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')

    parser.add_argument('--resume', action='store_true', default=False)

    parser.add_argument('--image_size', type=int, default=32,
                        help='size of image')
    parser.add_argument('--num_channels', type=int, default=12,
                        help='channel of wavelet subbands')
    parser.add_argument('--centered', action='store_false', default=True,
                        help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                        help='beta_max for diffusion')

    parser.add_argument('--patch_size', type=int, default=1,
                        help='Patchify image into non-overlapped patches')
    parser.add_argument('--num_channels_dae', type=int, default=128,
                        help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                        help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                        help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,), nargs='+', type=int,
                        help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                        help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                        help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                        help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                        help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                        help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                        help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                        help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    # generator and training

    parser.add_argument('--log_dir', default='./runs_new/', )
    parser.add_argument(
        '--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--datadir', default='./data')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int,
                        default=128, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float,
                        default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4,
                        help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 for adam')
    parser.add_argument('--no_lr_decay', action='store_true', default=False)

    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='use EMA or not')
    parser.add_argument('--ema_decay', type=float,
                        default=0.9999, help='decay rate for EMA')

    parser.add_argument('--r1_gamma', type=float,
                        default=5.0, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')
    parser.add_argument('--num_generator_updates', type=int, default=3,
                        help='Number of generator updates per discriminator update')

    # wavelet GAN
    parser.add_argument("--current_resolution", type=int, default=256)
    parser.add_argument("--use_pytorch_wavelet", action="store_true")
    parser.add_argument("--rec_loss", action="store_true")
    parser.add_argument("--net_type", default="normal")
    parser.add_argument("--num_disc_layers", default=6, type=int)
    parser.add_argument("--no_use_fbn", action="store_true")
    parser.add_argument("--no_use_freq", action="store_true")
    parser.add_argument("--no_use_residual", action="store_true")

    parser.add_argument('--save_content', action='store_true', default=True)
    parser.add_argument('--save_content_every', type=int, default=50,
                        help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int,
                        default=5, help='save ckpt every x epochs')

    # ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=3,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--master_port', type=str, default='6002',
                        help='port for master')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num_workers')

    #parser.add_argument('--summarywriter', default=None)

    args = parser.parse_args()

    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' %
                  (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(
                global_rank, global_size, train, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print('starting in debug mode')

        init_processes(0, size, train, args)
