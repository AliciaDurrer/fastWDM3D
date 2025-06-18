import torch
import lpips
import nibabel as nib
import numpy as np

# Load the LPIPS model
loss_fn = lpips.LPIPS(net='alex')  # You can also use 'vgg' or 'squeeze'

def calculate_lpips_in_mask(original, compressed, mask):
    # Ensure the mask is binary (0s and 1s)
    mask = mask.astype(np.float32)

    # Apply the mask to the original and compressed images
    masked_original = original * mask
    masked_compressed = compressed * mask

    # Convert to tensors and reshape for LPIPS calculation
    original_tensor = torch.tensor(masked_original).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, C, H, W)
    compressed_tensor = torch.tensor(masked_compressed).permute(2, 0, 1).unsqueeze(0)

    # Calculate LPIPS
    lpips_value = loss_fn(original_tensor, compressed_tensor).item()
    return lpips_value

# Load NIfTI images
original_nifti = nib.load('path_to_original_nifti')  # Replace with your NIfTI file path
compressed_nifti = nib.load('path_to_inpainted_nifti')  # Replace with your NIfTI file path

# Get the image data as numpy arrays
original = original_nifti.get_fdata()
compressed = compressed_nifti.get_fdata()

# Load or create a binary mask
mask_nifti = nib.load('path_to_mask_of_region_to_be_inpainted')  # Load mask as NIfTI
mask = mask_nifti.get_fdata()  # Get the mask data as a numpy array
mask = (mask > 0).astype(np.uint8)  # Convert to binary mask (0s and 1s)

# Calculate LPIPS in the masked region
lpips_value = calculate_lpips_in_mask(original, compressed, mask)

print(f"LPIPS in masked region: {lpips_value}")

