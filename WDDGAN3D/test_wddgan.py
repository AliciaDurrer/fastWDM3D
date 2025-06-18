import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
import numpy as np
import torch
import time
import math
import nibabel as nib

from diffusion import get_time_schedule, Posterior_Coefficients, \
    sample_from_model_test

from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D
from pytorch_wavelets import DWTInverse
from score_sde.models.ncsnpp_generator_adagn import NCSNpp, WaveletNCSNpp

from dataset import CreateDatasetSynthesisTest

from eval import create_submission_adapted
from eval import eval_sam

# %%
def sample_and_test(args):

    # Start time
    start_time = time.time()

    torch.manual_seed(args.seed)
    device = args.device

    save_dir = "./wddgan_generated_samples/{}_{}".format(args.dataset, args.checkpoint_path.split('/')[-1].split('.')[0])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logging.basicConfig(
        filename=os.path.join(save_dir, 'arguments_used.log'),
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    logging.info(f'args used: {args}')

    if args.dataset=='brats':
        dataset = CreateDatasetSynthesisTest(folder1=args.datadir)
        test_sampler = torch.utils.data.SequentialSampler(dataset)

        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  pin_memory=True,
                                                  sampler=test_sampler,
                                                  drop_last=True)
    elif args.dataset == 'cifar10':
        real_img_dir = 'pytorch_fid/cifar10_train_stat.npy'
    elif args.dataset == 'celeba_256':
        real_img_dir = 'pytorch_fid/celebahq_stat.npy'
    elif args.dataset == 'lsun':
        real_img_dir = 'pytorch_fid/lsun_church_stat.npy'
    else:
        real_img_dir = args.real_img_dir

    def to_range_0_1(x):
        return (x + 1.) / 2.

    args.ori_image_size = args.image_size
    args.image_size = args.current_resolution

    G_NET_ZOO = {"normal": NCSNpp, "wavelet": WaveletNCSNpp}
    gen_net = G_NET_ZOO[args.net_type]

    netG = gen_net(args).to(device)
    ckpt = torch.load('{}'.format(args.checkpoint_path), map_location=device)

    # loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)

    netG.load_state_dict(ckpt, strict=False)
    netG.eval()

    pytorch_total_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    logging.info(f'pytorch_total_params: {pytorch_total_params}')

    logging.info(f'network: {netG}')

    if not args.use_pytorch_wavelet:
        iwt = IDWT_3D("haar")
        dwt = DWT_3D("haar")
    else:
        iwt = DWTInverse(mode='zero', wave='haar').cuda()
    T = get_time_schedule(args, device)

    pos_coeff = Posterior_Coefficients(args, device)

    num_sam = 0

    start_actual_sampling = time.time()

    for iteration, (stacked_images, voided_image_full, starts, ends, labeled_masks, file_name) in enumerate(data_loader):

        voided_image_full = voided_image_full.to(device)

        LLL0, LLH0, LHL0, LHH0, HLL0, HLH0, HHL0, HHH0 = dwt(voided_image_full[:, :, :, :].unsqueeze(1))
        voided_image_full = torch.cat(
            [LLL0, LLH0, LHL0, LHH0, HLL0, HLH0, HHL0, HHH0], dim=1)
        voided_image_full = voided_image_full / math.sqrt(8.0)
        voided_image_full = voided_image_full.clamp(-1, 1)

        assert -1 <= voided_image_full.min() < 0
        assert 0 < voided_image_full.max() <= 1

        voided_image_full *= math.sqrt(8.0)
        voided_image_full = iwt(
            voided_image_full[:, 0:1], voided_image_full[:, 1:2], voided_image_full[:, 2:3], voided_image_full[:, 3:4],
            voided_image_full[:, 4:5],
            voided_image_full[:, 5:6], voided_image_full[:, 6:7], voided_image_full[:, 7:8])

        voided_image_full = torch.clamp(voided_image_full, -1, 1)
        voided_image_full = to_range_0_1(voided_image_full)  # 0-1

        voided_image_full = (voided_image_full - torch.min(voided_image_full) / torch.max(voided_image_full) - torch.min(voided_image_full))
        voided_image_full = voided_image_full.squeeze()

        for i, real_data in enumerate(stacked_images):

            real_data = torch.as_tensor(real_data, dtype=torch.float32)

            real_data = real_data.to(device)
            LLL0, LLH0, LHL0, LHH0, HLL0, HLH0, HHL0, HHH0 = dwt(real_data[:, 0, :, :, :].unsqueeze(1))
            LLL1, LLH1, LHL1, LHH1, HLL, HLH1, HHL1, HHH1 = dwt(real_data[:, 1, :, :, :].unsqueeze(1))
            real_data = torch.cat(
                [LLL0, LLH0, LHL0, LHH0, HLL0, HLH0, HHL0, HHH0, LLL1, LLH1, LHL1, LHH1, HLL, HLH1, HHL1, HHH1], dim=1)  # [b, 12, h, w]

            real_data = real_data / math.sqrt(8.0)  # [-1, 1]

            real_data = real_data.clamp(-1, 1)

            assert -1 <= real_data.min() < 0
            assert 0 < real_data.max() <= 1

            x_t_1 = torch.randn(args.batch_size, args.num_channels,
                                args.image_size, args.image_size, args.image_size).to(device)
            fake_sample = sample_from_model_test(
                pos_coeff, netG, args.num_timesteps, x_t_1, T, real_data, args)

            fake_sample *= math.sqrt(8.0)

            fake_sample = iwt(fake_sample[:, 0:1], args.enhancement_factor*fake_sample[:, 1:2], args.enhancement_factor*fake_sample[:, 2:3], args.enhancement_factor*fake_sample[:, 3:4], args.enhancement_factor*fake_sample[:, 4:5], args.enhancement_factor*fake_sample[:, 5:6], args.enhancement_factor*fake_sample[:, 6:7], args.enhancement_factor*fake_sample[:, 7:8])

            fake_sample = torch.clamp(fake_sample, -1, 1)
            fake_sample = to_range_0_1(fake_sample)  # 0-1

            fake_sample = fake_sample.squeeze()

            inpainted_image = voided_image_full.clone()

            start_idxs = np.asarray(starts[i])
            end_idxs = np.asarray(ends[i])

            s0, s1, s2 = start_idxs[0]
            e0, e1, e2 = end_idxs[0]

            inpainted_image[s0:e0, s1:e1, s2:e2] = 0
            inpainted_image[s0:e0, s1:e1, s2:e2] = fake_sample.squeeze()

            labeled_masks_i = labeled_masks[i].squeeze()

            voided_image_full[labeled_masks_i != -1] = inpainted_image[labeled_masks_i != -1]

        nib.save(nib.Nifti1Image(np.asarray(voided_image_full.cpu().detach().squeeze()), None),
                 os.path.join(save_dir, file_name[0].replace('t1n-voided', 't1n-inference')))

        num_sam+=1

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    print(f"Average time per sample (including loading): {elapsed_time/num_sam:.6f} seconds")

    elapsed_time_sampling_only = end_time - start_actual_sampling
    print(f"Elapsed time: {elapsed_time_sampling_only:.6f} seconds")
    print(f"Average time per sample (sampling only): {elapsed_time_sampling_only/num_sam:.6f} seconds")

    create_submission_adapted.adapt(input_data=args.datadir, samples_dir=save_dir, adapted_samples_dir=save_dir)
    print('adapted submissions saved in ', save_dir)
    #eval_sam.eval_adapted_samples(
    #    dataset_path_eval=args.datadir,
    #    solutionFilePaths_gt=args.datadir,
    #    resultsFolder_dir=save_dir, arguments=args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--enhancement_factor', type=float, default=1.0,
                        help='enhance higher frequency coefficients')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed used for initialization')
    parser.add_argument('--device', default='cuda:0',
                        help='GPU to use')
    parser.add_argument('--checkpoint_path', default='/home/user/model.pth')
    parser.add_argument('--log_dir', default='runs_test/')
    parser.add_argument('--datadir', default='data/test/')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                        help='whether or not compute FID')
    parser.add_argument('--measure_time', action='store_true', default=False,
                        help='whether or not measure time')
    parser.add_argument('--epoch_id', type=int, default=25)
    parser.add_argument('--num_channels', type=int, default=24,
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
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 2, 2, 2, 4],
                        help='channel multiplier')

    parser.add_argument('--num_res_blocks', type=int, default=5,
                        help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,), type=int, nargs='+',
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
    parser.add_argument(
        '--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy',
                        help='directory to real images for FID computation')

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32,
                        help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1,
                        help='sample generating batch size')

    # wavelet GAN
    parser.add_argument("--use_pytorch_wavelet", action="store_true")
    parser.add_argument("--current_resolution", type=int, default=256)
    parser.add_argument("--net_type", default="normal")
    parser.add_argument("--no_use_fbn", action="store_true")
    parser.add_argument("--no_use_freq", action="store_true")
    parser.add_argument("--no_use_residual", action="store_true")

    args = parser.parse_args()

    sample_and_test(args)
