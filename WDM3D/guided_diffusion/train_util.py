import copy
import functools
import os
import math
import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.utils.tensorboard
from torch.optim import AdamW
import torch.cuda.amp as amp

import itertools

from . import dist_util, logger
from .resample import LossAwareSampler, UniformSampler
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

INITIAL_LOG_LOSS_SCALE = 20.0


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        in_channels,
        image_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        resume_step,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        dataset="brats",
        summary_writer=None,
        mode="default",
        loss_level="image",
    ):
        self.summary_writer = summary_writer
        self.mode = mode
        self.model = model
        self.diffusion = diffusion
        self.datal = data
        self.dataset = dataset
        self.iterdatal = iter(data)
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.image_size = image_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        if self.use_fp16:
            self.grad_scaler = amp.GradScaler()
        else:
            self.grad_scaler = amp.GradScaler(enabled=False)

        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.dwt = DWT_3D("haar")
        self.idwt = IDWT_3D("haar")

        self.loss_level = loss_level

        self.step = 1
        self.resume_step = resume_step
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()

        self.opt = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            print("Resume Step: " + str(self.resume_step))
            self._load_optimizer_state()

        if not th.cuda.is_available():
            logger.warn("Training requires CUDA. ")

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print("resume model ...")
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)
        else:
            print("no optimizer checkpoint exists")

    def run_loop(self):
        import time

        t = time.time()
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            t_total = time.time() - t
            t = time.time()
            if self.dataset in ["brats", "brats3d", "lidc-idri"]:
                try:
                    batch, cond = next(self.iterdatal)
                # cond = {}
                except StopIteration:
                    self.iterdatal = iter(self.datal)
                    batch, cond = next(self.iterdatal)
                # cond = {}
            # Wavelet transform the data

            batch = th.cat((batch, cond), dim=1)

            batch = batch.to(dist_util.dev())

            cond = {}

            t_fwd = time.time()
            t_load = t_fwd - t

            lossmse, sample, sample_idwt, y_gt, y_masked, x_t, x_masked_dwt, x_mask_dwt, tused = self.run_step(batch, cond)
            
            x_t *= math.sqrt(8.0)
            B, _, H, W, D = x_t.size()
            x_t = self.idwt(
                x_t[:, 0, :, :, :].view(B, 1, H, W, D),
                x_t[:, 1, :, :, :].view(B, 1, H, W, D),
                x_t[:, 2, :, :, :].view(B, 1, H, W, D),
                x_t[:, 3, :, :, :].view(B, 1, H, W, D),
                x_t[:, 4, :, :, :].view(B, 1, H, W, D),
                x_t[:, 5, :, :, :].view(B, 1, H, W, D),
                x_t[:, 6, :, :, :].view(B, 1, H, W, D),
                x_t[:, 7, :, :, :].view(B, 1, H, W, D),
            )

            x_masked_dwt *= math.sqrt(8.0)
            B, _, H, W, D = x_masked_dwt.size()
            x_masked_dwt = self.idwt(
                x_masked_dwt[:, 0, :, :, :].view(B, 1, H, W, D),
                x_masked_dwt[:, 1, :, :, :].view(B, 1, H, W, D),
                x_masked_dwt[:, 2, :, :, :].view(B, 1, H, W, D),
                x_masked_dwt[:, 3, :, :, :].view(B, 1, H, W, D),
                x_masked_dwt[:, 4, :, :, :].view(B, 1, H, W, D),
                x_masked_dwt[:, 5, :, :, :].view(B, 1, H, W, D),
                x_masked_dwt[:, 6, :, :, :].view(B, 1, H, W, D),
                x_masked_dwt[:, 7, :, :, :].view(B, 1, H, W, D),
            )

            x_mask_dwt *= math.sqrt(8.0)
            B, _, H, W, D = x_mask_dwt.size()
            x_mask_dwt = self.idwt(
                x_mask_dwt[:, 0, :, :, :].view(B, 1, H, W, D),
                x_mask_dwt[:, 1, :, :, :].view(B, 1, H, W, D),
                x_mask_dwt[:, 2, :, :, :].view(B, 1, H, W, D),
                x_mask_dwt[:, 3, :, :, :].view(B, 1, H, W, D),
                x_mask_dwt[:, 4, :, :, :].view(B, 1, H, W, D),
                x_mask_dwt[:, 5, :, :, :].view(B, 1, H, W, D),
                x_mask_dwt[:, 6, :, :, :].view(B, 1, H, W, D),
                x_mask_dwt[:, 7, :, :, :].view(B, 1, H, W, D),
            )

            x_t = torch.clamp(x_t, -1, 1)
            x_masked_dwt = torch.clamp(x_masked_dwt, -1, 1)
            x_mask_dwt = torch.clamp(x_mask_dwt, -1, 1)

            x_t = (x_t + 1.) / 2.
            x_masked_dwt = (x_masked_dwt + 1.) / 2.
            x_mask_dwt = (x_mask_dwt + 1.) / 2.

            t_fwd = time.time() - t_fwd

            if self.summary_writer is not None:
                self.summary_writer.add_scalar(
                    "time/load", t_load, global_step=self.step + self.resume_step
                )
                self.summary_writer.add_scalar(
                    "time/forward", t_fwd, global_step=self.step + self.resume_step
                )
                self.summary_writer.add_scalar(
                    "time/total", t_total, global_step=self.step + self.resume_step
                )
                self.summary_writer.add_scalar(
                    "loss/rec", lossmse.item(), global_step=self.step + self.resume_step
                )


            if self.step % 100 == 0:

               y_masked = (torch.clamp(y_masked, -1, 1) + 1) / 2

               self.summary_writer.add_scalar(
                       "t_used/t",
                       tused,
                       global_step=self.step + self.resume_step,
                   )

               self.summary_writer.add_image(
                       "sample/sample_idwt_0",
                        sample_idwt[0, :, 64, :, :],
                        global_step=self.step + self.resume_step,
                    )

               self.summary_writer.add_image(
                    "sample/sample_idwt_1",
                    sample_idwt[0, :, :, 64, :],
                    global_step=self.step + self.resume_step,
                )
               self.summary_writer.add_image(
                       "sample/sample_idwt_2",
                        sample_idwt[0, :, :, :, 64],
                        global_step=self.step + self.resume_step,
                    )
               self.summary_writer.add_image(
                   "gt/gt_0",
                   y_gt[0, :, 64, :, :],
                   global_step=self.step + self.resume_step,
               )

               self.summary_writer.add_image(
                   "gt/gt_1",
                   y_gt[0, :, :, 64, :],
                   global_step=self.step + self.resume_step,
               )
               self.summary_writer.add_image(
                   "gt/gt_2",
                   y_gt[0, :, :, :, 64],
                   global_step=self.step + self.resume_step,
               )

               self.summary_writer.add_image(
                    "voided/voided_0",
                    y_masked[0, :, 64, :, :],
                    global_step=self.step + self.resume_step,
                )

               self.summary_writer.add_image(
                    "voided/voided_1",
                    y_masked[0, :, :, 64, :],
                    global_step=self.step + self.resume_step,
                )
               self.summary_writer.add_image(
                    "voided/voided_2",
                    y_masked[0, :, :, :, 64],
                    global_step=self.step + self.resume_step,
                )
                
               self.summary_writer.add_image(
                    "x_t/x_t_0",
                    x_t[0, :, 64, :, :],
                    global_step=self.step + self.resume_step,
                )

               self.summary_writer.add_image(
                    "x_t/x_t_1",
                    x_t[0, :, :, 64, :],
                    global_step=self.step + self.resume_step,
                )
               self.summary_writer.add_image(
                    "x_t/x_t_2",
                    x_t[0, :, :, :, 64],
                    global_step=self.step + self.resume_step,
                )
                
               self.summary_writer.add_image(
                    "x_masked_dwt/x_masked_dwt_0",
                    x_masked_dwt[0, :, 64, :, :],
                    global_step=self.step + self.resume_step,
                )

               self.summary_writer.add_image(
                    "x_masked_dwt/x_masked_dwt_1",
                    x_masked_dwt[0, :, :, 64, :],
                    global_step=self.step +self.resume_step,
                )
               self.summary_writer.add_image(
                    "x_masked_dwt/x_masked_dwt_2",
                    x_masked_dwt[0, :, :, :, 64],
                    global_step=self.step + self.resume_step,
                )
                
               self.summary_writer.add_image(
                    "x_mask_dwt/x_mask_dwt_0",
                    x_mask_dwt[0, :, 64, :, :],
                    global_step=self.step + self.resume_step,
                )

               self.summary_writer.add_image(
                    "x_mask_dwt/x_mask_dwt_1",
                    x_mask_dwt[0, :, :, 64, :],
                    global_step=self.step + self.resume_step,
                )
               self.summary_writer.add_image(
                    "x_mask_dwt/x_mask_dwt_2",
                    x_mask_dwt[0, :, :, :, 64],
                    global_step=self.step + self.resume_step,
                )

            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, label=None, info=dict()):
        lossmse, sample, sample_idwt, y_gt, y_masked, x_t, x_masked_dwt, x_mask_dwt, tused = self.forward_backward(batch, cond, label)

        if self.use_fp16:
            self.grad_scaler.unscale_(
                self.opt
            )  # check self.grad_scaler._per_optimizer_states

        # compute norms
        with torch.no_grad():
            param_max_norm = max(
                [p.abs().max().item() for p in self.model.parameters()]
            )
            grad_max_norm = max(
                [p.grad.abs().max().item() for p in self.model.parameters()]
            )
            info["norm/param_max"] = param_max_norm
            info["norm/grad_max"] = grad_max_norm

        if not torch.isfinite(lossmse):  # infinite
            if not torch.isfinite(torch.tensor(param_max_norm)):
                logger.log(
                    f"Model parameters contain non-finite value {param_max_norm}, entering breakpoint",
                    level=logger.ERROR,
                )
                breakpoint()
            else:
                logger.log(
                    f"Model parameters are finite, but loss is not: {lossmse}"
                    "\n -> update will be skipped in grad_scaler.step()",
                    level=logger.WARN,
                )

        if self.use_fp16:
            print("Use fp16 ...")
            self.grad_scaler.step(self.opt)
            self.grad_scaler.update()
            info["scale"] = self.grad_scaler.get_scale()
        else:
            self.opt.step()
        self._anneal_lr()
        self.log_step()
        return lossmse, sample, sample_idwt, y_gt, y_masked, x_t, x_masked_dwt, x_mask_dwt, tused

    def forward_backward(self, batch, cond, label=None):
        for p in self.model.parameters():  # Zero out gradient
            p.grad = None

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())

            if label is not None:
                micro_label = label[i : i + self.microbatch].to(dist_util.dev())
            else:
                micro_label = None

            micro_cond = None

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                x_start=micro,
                t=t,
                model_kwargs=micro_cond,
                labels=micro_label,
                mode=self.mode,  # 'default' (image generation) or 'segmentation'
                loss_level=self.loss_level,
            )

            losses1 = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses1["loss"].detach()
                )

            loss = losses1[0]
            sample = losses1[1]  # Denoised subbands at t=0
            sample_idwt = losses1[2]  # Inverse wavelet transformed prediction at t=0
            y_gt = losses1[3]
            y_masked = losses1[4]
            x_t = losses1[5]
            x_masked_dwt = losses1[6]
            x_mask_dwt = losses1[7]
            tused = losses1[8]

            lossret = loss.detach()

            self.summary_writer.add_scalar(
                 "loss/rec_loss_full_plus_masked",
                 loss.item(),
                 global_step=self.step + self.resume_step,
            )

            # perform some finiteness checks
            if not torch.isfinite(loss):
                logger.log(f"Encountered non-finite loss {loss}")
            if self.use_fp16:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            return lossret.detach(), sample, sample_idwt, y_gt, y_masked, x_t, x_masked_dwt, x_mask_dwt, tused

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, state_dict):
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if self.dataset == "brats":
                    filename = f"bratsimage{(self.step+self.resume_step):06d}.pt"
                elif self.dataset == "brats3d":
                    filename = f"brats3dimage{(self.step+self.resume_step):06d}.pt"
                elif self.dataset == "chexpert":
                    filename = f"cheximage{(self.step + self.resume_step):06d}.pt"
                elif self.dataset == "lidc-idri":
                    filename = f"lidcimage{(self.step+self.resume_step):06d}.pt"
                else:
                    raise ValueError(f"dataset {self.dataset} not implemented")

                print("filename", filename)
                with bf.BlobFile(
                    bf.join(get_blob_logdir(), "checkpoints", filename), "wb"
                ) as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.model.state_dict())

        if dist.get_rank() == 0:
            checkpoint_dir = os.path.join(logger.get_dir(), "checkpoints")
            with bf.BlobFile(
                bf.join(checkpoint_dir, f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)
        print("saved opt checkpoint")
        dist.barrier()  # stop training when save_interval >  1


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """

    split = os.path.basename(filename)
    split = split.split(".")[-2]  # remove extension
    split = split.split("_")[-1]  # remove possible underscores, keep only last word
    # extract trailing number
    reversed_split = []
    for c in reversed(split):
        if not c.isdigit():
            break
        reversed_split.append(c)
    split = "".join(reversed(reversed_split))
    split = "".join(c for c in split if c.isdigit())  # remove non-digits
    try:
        return int(split)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
