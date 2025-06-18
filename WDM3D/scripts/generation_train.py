"""
Train a diffusion model to generate images.
"""
import os
import sys
import argparse
import torch as th
import random
import numpy as np
import shutil

sys.path.append("..")
sys.path.append(".")

from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.lidcloader import LIDCVolumes
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from torch.utils.tensorboard import SummaryWriter

def main():
    args = create_argparser().parse_args()
    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    summary_writer = None
    if args.use_tensorboard:
        logdir = None
        if args.tensorboard_path:
            logdir = args.tensorboard_path
        summary_writer = SummaryWriter(log_dir=logdir)
        summary_writer.add_text(
            "config",
            "\n".join([f"--{k}={repr(v)} <br/>" for k, v in vars(args).items()]),
        )
        print(
            f"[TENSORBOARD] Using Tensorboard with logdir = {summary_writer.get_logdir()}"
        )
        logger.configure(dir=summary_writer.get_logdir())
    else:
        logger.configure()

    dist_util.setup_dist(devices=args.devices)

    logdir_files = os.path.join(summary_writer.get_logdir(), "files/")
    if not os.path.exists(logdir_files):
        os.makedirs(logdir_files)
    shutil.copytree('/path/to/folder/WDM_3D/scripts/', os.path.join(logdir_files, 'scripts'))
    shutil.copytree('/path/to/folder/WDM_3D/guided_diffusion/', os.path.join(logdir_files, 'guided_diffusion'))
    shutil.copyfile('/path/to/folder/WDM_3D/run.sh', os.path.join(logdir_files, 'run.sh'))

    logger.log("[INFO] Creating model and diffusion...")
    logger.log("[ARGS] ", args)
    arguments = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**arguments)

    print(
        "[MODEL] Number of trainable parameters: {}".format(
            np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
        )
    )
    model.to(
        dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev()
    )  # allow for 2 devices
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion, maxt=args.diffusion_steps
    )
    
    logger.log('model: ', model)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.log(f'number of model parameters: {pytorch_total_params}')

    logger.log("[LOGGER] Creating data loader...")

    if args.dataset == "brats3d":
        assert args.image_size in [64, 128, 256]
        print(args.data_dir)
        ds = BRATSVolumes(
            args.data_dir
        )

        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )

    if args.dataset == "lidc-idri":
        assert args.image_size in [64, 128, 256]
        print(args.data_dir)
        ds = LIDCVolumes(
            args.data_dir,
            test_flag=False,
            normalize=(lambda x: 2 * x - 1) if args.renormalize else None,
            mode="train",
            img_size=args.image_size,
        )

        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )

    print(args.resume_checkpoint)
    logger.log("[TRAINING] Start training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=datal,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        image_size=args.image_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        resume_step=args.resume_step,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset,
        summary_writer=summary_writer,
        mode="default",
    ).run_loop()


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        beta_min=0.1,
        beta_max=20.0,
        ema_rate="0.9999",
        log_interval=1000,
        save_interval=10000,
        resume_checkpoint="",
        resume_step=0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset="brats3d",
        use_tensorboard=True,
        tensorboard_path="",  # set path to existing logdir for resuming
        devices=[0],
        dims=3,  # 2 for 2d images, 3 for 3d volumes
        learn_sigma=False,
        num_groups=29,
        channel_mult="1,2,2,4",
        in_channels=8,
        out_channels=8,
        bottleneck_attention=False,
        num_workers=0,
        mode="default",
        renormalize=True,
        additive_skips=False,
        use_freq=False,
        use_wgupdown=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
