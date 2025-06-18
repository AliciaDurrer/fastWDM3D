"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import sys
import random

sys.path.append(".")
import numpy as np
import math
import time
import torch as th
import torch.distributed as dist
import nibabel as nib
import pathlib
import warnings
from datetime import datetime
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BraTSVolumesTest
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
#from diffusion import get_time_schedule, Posterior_Coefficients, \
#    sample_from_model_test

from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

from eval import create_submission_adapted
from eval import eval_sam

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


def dice_score(pred, targs):
    pred = (pred > 0).float()
    return 2.0 * (pred * targs).sum() / (pred + targs).sum()

def to_range_0_1(x):
    return (x + 1.) / 2.


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    dist_util.setup_dist(devices=args.devices)
    save_dir = os.path.join(args.output_dir, args.model_path.split('/')[-1].split('.')[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger.configure(dir=save_dir)

    start_time = time.time()

    logger.log("[INFO] creating model and diffusion...")
    logger.log("[ARGS] ", args)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    logger.log("[INFO] load model from: {}".format(args.model_path))
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    logger.log("[MODEL] ", model)
    model.to(
        dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev()
    )  # allow for 2 devices

    if args.use_fp16:
        raise ValueError("fp16 currently not implemented")

    model.eval()
    idwt = IDWT_3D("haar")
    dwt = DWT_3D("haar")

    dataset = BraTSVolumesTest(folder1=args.data_dir)
    test_sampler = th.utils.data.SequentialSampler(dataset)

    data_loader = th.utils.data.DataLoader(dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=4,
                                           pin_memory=True,
                                           sampler=test_sampler,
                                           drop_last=True)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    num_sam = 0

    start_actual_sampling = time.time()

    for iteration, (stacked_images, voided_image_full, starts, ends, labeled_masks, file_name) in enumerate(
            data_loader):

        logger.log("sampling file ", str(file_name))
        logger.log("    sampling start ", datetime.now())

        th.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"reseeded (in for loop) to {seed}")

        seed += 1

        voided_image_full = voided_image_full.to(dist_util.dev())

        LLL0, LLH0, LHL0, LHH0, HLL0, HLH0, HHL0, HHH0 = dwt(voided_image_full[:, :, :, :].unsqueeze(1))
        voided_image_full = th.cat(
            [LLL0, LLH0, LHL0, LHH0, HLL0, HLH0, HHL0, HHH0], dim=1)
        voided_image_full = voided_image_full / math.sqrt(8.0)
        voided_image_full = voided_image_full.clamp(-1, 1)

        assert -1 <= voided_image_full.min() < 0
        assert 0 < voided_image_full.max() <= 1

        voided_image_full *= math.sqrt(8.0)
        voided_image_full = idwt(
            voided_image_full[:, 0:1], voided_image_full[:, 1:2], voided_image_full[:, 2:3], voided_image_full[:, 3:4],
            voided_image_full[:, 4:5],
            voided_image_full[:, 5:6], voided_image_full[:, 6:7], voided_image_full[:, 7:8])

        voided_image_full = th.clamp(voided_image_full, -1, 1)
        voided_image_full = to_range_0_1(voided_image_full)  # 0-1

        voided_image_full = (voided_image_full - th.min(voided_image_full) / th.max(voided_image_full) - th.min(
            voided_image_full))
        voided_image_full = voided_image_full.squeeze()

        for i, real_data in enumerate(stacked_images):
            input_masked = real_data[0, 0, :, :, :].to(dist_util.dev())
            input_masked = input_masked.unsqueeze(0).unsqueeze(1)

            mask = real_data[0, 1, :, :, :].to(dist_util.dev())
            mask = mask.unsqueeze(0).unsqueeze(1)

            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(input_masked)
            input_masked_dwt = th.cat([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(mask)
            mask_dwt = th.cat([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

            input_masked_dwt = input_masked_dwt / math.sqrt(8.0)
            mask_dwt = mask_dwt / math.sqrt(8.0)

            input_masked_dwt = input_masked_dwt.clamp(-1, 1)
            mask_dwt = mask_dwt.clamp(-1, 1)

            img = th.randn(
                args.batch_size,
                8,
                args.image_size // 2,
                args.image_size // 2,
                args.image_size // 2,
            ).to(dist_util.dev())

            model_kwargs = {}

            sample_fn = diffusion.p_sample_loop

            sample = sample_fn(
                t=args.sampling_steps,
                model=model,
                shape=img.shape,
                noise=img,
                input_masked=input_masked_dwt,
                mask=mask_dwt,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            fake_sample = sample * math.sqrt(8.0)

            fake_sample = idwt(fake_sample[:, 0:1], fake_sample[:, 1:2],
                               fake_sample[:, 2:3],
                               fake_sample[:, 3:4],
                               fake_sample[:, 4:5],
                               fake_sample[:, 5:6],
                               fake_sample[:, 6:7],
                               fake_sample[:, 7:8])

            fake_sample = th.clamp(fake_sample, -1, 1)
            fake_sample = to_range_0_1(fake_sample)  # 0-1

            fake_sample = fake_sample.squeeze()

            inpainted_image = voided_image_full.clone()

            start_idxs = np.asarray(starts[i])
            end_idxs = np.asarray(ends[i])

            s0, s1, s2 = start_idxs[0]
            e0, e1, e2 = end_idxs[0]

            inpainted_image[s0:e0, s1:e1, s2:e2] = 0
            inpainted_image[s0:e0, s1:e1, s2:e2] = fake_sample

            labeled_masks_i = labeled_masks[i].squeeze()

            voided_image_full[labeled_masks_i != -1] = inpainted_image[labeled_masks_i != -1]

        nib.save(nib.Nifti1Image(np.asarray(voided_image_full.cpu().detach().squeeze()), None),
                 os.path.join(save_dir, file_name[0].replace('t1n-voided', 't1n-inference')))

        num_sam += 1

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    logger.info(f"Average time per sample (including loading): {elapsed_time / num_sam:.6f} seconds")

    elapsed_time_sampling_only = end_time - start_actual_sampling
    print(f"Elapsed time: {elapsed_time_sampling_only:.6f} seconds")
    logger.info(f"Average time per sample (sampling only): {elapsed_time_sampling_only / num_sam:.6f} seconds")

    print('preparing samples for evaluation ...')
    create_submission_adapted.adapt(input_data=args.data_dir, samples_dir=save_dir, adapted_samples_dir=save_dir)
    print('samples ready for evaluation, saved in ', save_dir)
    #eval_sam.eval_adapted_samples(
    #    dataset_path_eval=args.data_dir,
    #    solutionFilePaths_gt=args.data_dir,
    #    resultsFolder_dir=save_dir)


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        data_mode="validation",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        model_path="",
        devices=[0],
        output_dir="./results",
        mode="default",
        renormalize=False,
        image_size=256,
        half_res_crop=False,
        concat_coords=False,  # if true, add 3 (for 3d) or 2 (for 2d) to in_channels
    )
    defaults.update(
        {k: v for k, v in model_and_diffusion_defaults().items() if k not in defaults}
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
