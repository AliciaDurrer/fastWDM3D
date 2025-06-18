# Based on https://github.com/BraTS-inpainting/inpainting.git

import sys
from tqdm import tqdm
from pathlib import Path
import nibabel as nib

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import MeanSquaredError
import torch
import numpy as np

# Define evaluation Metrics
psnr_01 = PeakSignalNoiseRatio(data_range=1.0)  # because we normalize to 0-1
psnr = PeakSignalNoiseRatio()  # default versionen that uses max
ssim = StructuralSimilarityIndexMeasure(return_full_image=True)
mse = MeanSquaredError()


def __percentile_clip(
    input_tensor, reference_tensor=None, p_min=0.5, p_max=99.5, strictlyPositive=True
):
    """Normalizes a tensor based on percentiles. Clips values below and above the percentile.
    Percentiles for normalization can come from another tensor.

    Args:
        input_tensor (torch.Tensor): Tensor to be normalized based on the data from the reference_tensor.
            If reference_tensor is None, the percentiles from this tensor will be used.
        reference_tensor (torch.Tensor, optional): The tensor used for obtaining the percentiles.
        p_min (float, optional): Lower end percentile. Defaults to 0.5.
        p_max (float, optional): Upper end percentile. Defaults to 99.5.
        strictlyPositive (bool, optional): Ensures that really all values are above 0 before normalization. Defaults to True.

    Returns:
        torch.Tensor: The input_tensor normalized based on the percentiles of the reference tensor.
    """
    if reference_tensor == None:
        reference_tensor = input_tensor
    v_min, v_max = np.percentile(
        reference_tensor, [p_min, p_max]
    )  # get p_min percentile and p_max percentile

    if v_min < 0 and strictlyPositive:  # set lower bound to be 0 if it would be below
        v_min = 0
    output_tensor = np.clip(
        input_tensor, v_min, v_max
    )  # clip values to percentiles from reference_tensor
    output_tensor = (output_tensor - v_min) / (
        v_max - v_min
    )  # normalizes values to [0;1]

    return output_tensor


def compute_metrics(
    gt_image: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor, normalize=True
):
    """Computes MSE, PSNR and SSIM between two images only in the masked region.

    Normalizes the two images to [0;1] based on the gt_image 0.5 and 99.5 percentile in the non-masked region.
    Requires input to have shape (1,1, X,Y,Z), meaning only one sample and one channel.
    For MSE and PSNR we use the respective torchmetrics libraries on the voxels that are covered by the mask.
    For SSIM, we first zero all non-mask voxels, then we apply regular SSIM on the complete volume. In the end we take
    the "full SSIM" image from torchmetrics and only take the values relating to voxels within the mask.
    The main difference between the original torchmetrics SSIM and this substitude for masked images is that we pad
    with zeros while torchmetrics does reflection padding at the cuboid borders.
    This does slightly bias the SSIM voxel values at the mask surface but does not influence the resulting participant
    ranking as all submission underlie the same bias.

    Args:
        gt_image (torch.Tensor): The t1n ground truth image (t1n.nii.gz)
        prediction (torch.Tensor): The inferred/predicted t1n image
        mask (torch.Tensor): The inference mask (mask.nii.gz)
        normalize (bool): Normalizes the input by dividing trough the maximal value of the gt_image in the masked
            region. Defaults to True

    Raises:
        UserWarning: If you dimensions do not match the (torchmetrics) requirements: 1,1,X,Y,Z

    Returns:
        float: (MSE, PSNR, SSIM)
    """

    if not (prediction.shape[0] == 1 and prediction.shape[1] == 1):
        raise UserWarning(
            f"All inputs have to be 5D with the first two dimensions being 1. Your prediction dimension: {prediction.shape}"
        )

    # Get Infill region (we really are only interested in the infill region)
    prediction_infill = prediction * mask
    gt_image_infill = gt_image * mask

    # Normalize to [0;1] based on GT (otherwise MSE will depend on the image intensity range)
    if normalize:
        reference_tensor = (
            gt_image * ~mask
        )  # use all the tissue that is not masked for normalization
        gt_image_infill = __percentile_clip(
            gt_image_infill,
            reference_tensor=reference_tensor,
            p_min=0.5,
            p_max=99.5,
            strictlyPositive=True,
        )
        prediction_infill = __percentile_clip(
            prediction_infill,
            reference_tensor=reference_tensor,
            p_min=0.5,
            p_max=99.5,
            strictlyPositive=True,
        )

    # SSIM - apply on complete masked image but only take values from masked region
    full_cuboid_SSIM, ssim_idx_full_image = ssim(
        preds=prediction_infill, target=gt_image_infill
    )
    ssim_idx = ssim_idx_full_image[mask]
    SSIM = ssim_idx.mean()

    # only voxels that are to be inferred (-> flat array)
    gt_image_infill = gt_image_infill[mask]
    prediction_infill = prediction_infill[mask]

    # MSE
    MSE = mse(preds=prediction_infill, target=gt_image_infill)

    # PSNR with fixed data range (using top down knowledge of our data. Namely that it is always in the range of 0 to 1)
    PSNR_01 = psnr_01(preds=prediction_infill, target=gt_image_infill)

    # PSNR
    PSNR = psnr(preds=prediction_infill, target=gt_image_infill)

    return float(MSE), float(PSNR), float(PSNR_01), float(SSIM)


def eval_adapted_samples(dataset_path_eval=None, solutionFilePaths_gt=None, resultsFolder_dir=None, arguments=None):
    o = sys.stdout

    result_folder_name = resultsFolder_dir.split('/')[-1]

    with open(f'sampling_output_scores/{result_folder_name}.txt', 'w') as f:
        sys.stdout = f

        print('args: ', arguments)

        # Task dataset (on synapse server)
        dataset_path = Path(str(dataset_path_eval))
        dataset_path_prefix = str(dataset_path_eval)
        solutionFilePaths_gt = Path(str(solutionFilePaths_gt))
        solutionFilePaths = sorted(list(solutionFilePaths_gt.rglob("**/BraTS-GLI-*-*-t1n.nii.gz")))
        print(f"Task: {dataset_path}")
        print(f"\tlen: {len(solutionFilePaths)}")

        # Solution dataset (participant upload)
        resultsFolder = Path(str(resultsFolder_dir))
        resultFilePaths = sorted(list(resultsFolder.rglob("**/BraTS_*-*.nii.gz")))
        print(f"Solution: {resultsFolder}")
        print(f"\tlen: {len(resultFilePaths)}")

        i = 0

        # Evaluation
        performance = {"folderName": [], "SSIM": [], "PSNR": [], "MSE": []}

        print(f"FolderName\t\tSSIM\t\tPSNR\t\tMSE")
        for resultFilePath in sorted(resultsFolder.rglob("**/*.nii.gz")):
            folderName = resultFilePath.name[:19]
            folderPath = dataset_path.joinpath(folderName)
            gtPath = solutionFilePaths_gt.joinpath(folderName)
            if gtPath.exists() == False:
                print(f'Result with ID "{folderName}" has no corresponding solution folder {gtPath}')
            else:
                performance["folderName"].append(folderName)
                # Read result
                result_img = nib.load(resultFilePath)
                result = torch.Tensor(result_img.get_fdata()).unsqueeze(0).unsqueeze(0)
                # Inference mask
                mask_path = folderPath.joinpath(f"{folderName}-mask-healthy.nii.gz")
                mask_img = nib.load(mask_path)
                mask = torch.Tensor(mask_img.get_fdata()).bool().unsqueeze(0).unsqueeze(0)
                # Ground truth
                t1n_path = solutionFilePaths_gt.joinpath(folderName).joinpath(f"{folderName}-t1n.nii.gz")
                t1n_img = nib.load(t1n_path)
                t1n = torch.Tensor(t1n_img.get_fdata()).unsqueeze(0).unsqueeze(0)
                # Compute metrics
                sum_diff = torch.sum(torch.abs(t1n - result))
                np_t1n = np.asarray(t1n.squeeze(0).squeeze(0))
                np_result = np.asarray(result.squeeze(0).squeeze(0))

                # Compute metrics
                MSE, PSNR, PSNR_01, SSIM = compute_metrics(gt_image=t1n, prediction=result, mask=mask)
                # Scores
                performance["SSIM"].append(SSIM)
                performance["PSNR"].append(PSNR_01)  # use PSNR with fixed data range as default.
                performance["MSE"].append(MSE)

                print(
                    f'{folderName}\t{performance["SSIM"][-1]:.8f}\t{performance["PSNR"][-1]:.8f}\t{performance["MSE"][-1]:.8f}')

                i += 1

                if i == 51:
                    break

        print('average SSIM', np.average(performance["SSIM"]))
        print('average PSNR', np.average(performance["PSNR"]))
        print('average MSE', np.average(performance["MSE"]))
        print('std SSIM', np.std(performance["SSIM"]))
        print('std PSNR', np.std(performance["PSNR"]))
        print('std MSE', np.std(performance["MSE"]))

    sys.stdout = o

