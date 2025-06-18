import os
import torch
import torch.utils.data
import numpy as np
import nibabel as nib
from skimage.measure import label


class CreateDatasetSynthesis(torch.utils.data.Dataset):
    def __init__(self, folder1):
        self.patients = sorted([d for d in os.listdir(folder1) if os.path.isdir(os.path.join(folder1, d))])
        self.folder1 = folder1

        self.labels = [0] * len(self.patients)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_folder = os.path.join(self.folder1, self.patients[idx])

        mask_name = self.patients[idx] + '-mask-healthy.nii.gz'
        t1n_name = self.patients[idx] + '-t1n.nii.gz'
        mask_file = os.path.join(patient_folder, mask_name)
        t1n_file = os.path.join(patient_folder, t1n_name)

        # Load mask
        mask_image = self.load_data(mask_file)
        mask_image[mask_image != 1] = -1
        mask_image = np.pad(mask_image, ((8, 8), (8, 8), (50, 51)), 'constant', constant_values=-1)
        mask_image, start, end = self.crop_mask_to_center(mask_image)

        if mask_image.shape != (128, 128, 128):
            raise ValueError("Generated mask {} must be of shape (128, 128, 128) but is {}!".format(mask_file, mask_image.shape))
        mask_image = torch.as_tensor(mask_image, dtype=torch.float32)
        
        t1n_image = self.load_data(t1n_file)
        t1n_image = np.clip(
            t1n_image,
            np.quantile(t1n_image, 0.005),
            np.quantile(t1n_image, 0.995),
        )
        t1n_image = 2 * (t1n_image - np.min(t1n_image)) / (np.max(t1n_image) - np.min(t1n_image)) - 1
        t1n_image = np.pad(t1n_image, ((8, 8), (8, 8), (50, 51)), 'constant', constant_values=-1)
        t1n_image = torch.as_tensor(t1n_image, dtype=torch.float32)
        t1n_image = t1n_image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        mask_idx = mask_image.clone()

        voided_image = t1n_image.clone()
        voided_image[mask_image == 1] = -1

        t1n_image = t1n_image.unsqueeze(0)
        mask_image = mask_image.unsqueeze(0)
        voided_image = voided_image.unsqueeze(0)

        stacked_image = torch.cat((voided_image, mask_image, t1n_image), dim=0)

        label = self.labels[idx]

        return stacked_image, label, mask_idx

    def crop_mask_to_center(self, mask):
        if mask.shape != (256, 256, 256):
            raise ValueError("Input mask must be of shape (256, 256, 256)")

        indices = np.argwhere(mask == 1)

        if indices.size == 0:
            raise ValueError("The mask does not contain any 1's.")

        centroid = np.mean(indices, axis=0).astype(int)

        # Define the cropping indices
        start = centroid - 64  # 128 / 2 = 64
        end = centroid + 64  # 128 / 2 = 64
        # Ensure the cropping indices are within bounds and maintain a size of 128
        for i in range(3):
            if start[i] < 0:
                start[i] = 0
                end[i] = 128
            elif end[i] > 256:
                end[i] = 256
                start[i] = 256 - 128
        # Ensure the difference between start and end is always 128
        for i in range(3):
            if end[i] - start[i] != 128:
                if end[i] - start[i] < 128:
                    # If the range is less than 128, adjust the start
                    start[i] = end[i] - 128
                else:
                    # If the range is more than 128, adjust the end
                    end[i] = start[i] + 128
        # Crop the mask to the center
        cropped_mask = mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        return cropped_mask, start, end

    def load_data(self, load_dir):
        img = nib.load(load_dir)
        data = img.get_fdata()

        data = data.astype(np.float32)

        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)
            data = np.expand_dims(data, axis=1)

        return np.asarray(data, dtype=np.float32)


class CreateDatasetSynthesisTest(torch.utils.data.Dataset):
    def __init__(self, folder1):
        self.patients = sorted([d for d in os.listdir(folder1) if os.path.isdir(os.path.join(folder1, d))])
        self.folder1 = folder1

        self.labels = [0] * len(self.patients)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_folder = os.path.join(self.folder1, self.patients[idx])

        mask_name = self.patients[idx] + '-mask.nii.gz'
        t1n_name = self.patients[idx] + '-t1n-voided.nii.gz'
        mask_file = os.path.join(patient_folder, mask_name)
        t1n_file = os.path.join(patient_folder, t1n_name)

        mask_image = self.load_data(mask_file)
        mask_image[mask_image != 1] = -1
        mask_image = np.pad(mask_image, ((8, 8), (8, 8), (50, 51)), 'constant', constant_values=-1)
        cropped_masks, starts, ends, labeled_masks = self.crop_mask_to_center(mask_image)

        t1n_image = self.load_data(t1n_file)
        t1n_image = torch.as_tensor(t1n_image, dtype=torch.float32)
        t1n_image = np.clip(
            t1n_image,
            np.quantile(t1n_image, 0.005),
            np.quantile(t1n_image, 0.995),
        )
        t1n_image = 2 * (t1n_image - torch.min(t1n_image)) / (torch.max(t1n_image) - torch.min(t1n_image)) - 1
        t1n_image = torch.tensor(np.pad(t1n_image, ((8, 8), (8, 8), (50, 51)), 'constant', constant_values=-1))

        stacked_images = []
        for cropped_mask, start, end, labeled_mask in zip(cropped_masks, starts, ends, labeled_masks):
            cropped_mask_tensor = torch.as_tensor(cropped_mask, dtype=torch.float32)

            voided_image = t1n_image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

            voided_image_full = t1n_image.clone()

            stacked_image = torch.cat((voided_image.unsqueeze(0), cropped_mask_tensor.unsqueeze(0)), dim=0)
            stacked_images.append(stacked_image)

        return stacked_images, voided_image_full, starts, ends, labeled_masks, t1n_name

    def crop_mask_to_center(self, mask):
        if mask.shape != (256, 256, 256):
            raise ValueError("Input mask must be of shape (256, 256, 256)")

        # Find connected components in the mask using skimage
        labeled_mask = label(mask, background=-1, connectivity=3)  # Use 3D connectivity
        num_features = labeled_mask.max()  # The maximum label value corresponds to the number of features

        if num_features == 0:  # No components found
            raise ValueError("The mask does not contain any 1's.")

        cropped_masks = []
        starts = []
        ends = []
        labeled_masks = []

        for i in range(1, num_features + 1):  # Iterate over each component (skip the background label 0)
            # Get the indices of the current component
            indices = np.argwhere(labeled_mask == i)
            labeled_mask_copy = labeled_mask.copy()
            labeled_mask_copy[labeled_mask_copy!=i] = -1

            # Calculate the centroid of the component
            centroid = np.mean(indices, axis=0).astype(int)

            # Define the cropping indices
            start = centroid - 64  # 128 / 2 = 64
            end = centroid + 64  # 128 / 2 = 64
            # Ensure the cropping indices are within bounds and maintain a size of 128
            for j in range(3):
                if start[j] < 0:
                    start[j] = 0
                    end[j] = 128
                elif end[j] > 256:
                    end[j] = 256
                    start[j] = 256 - 128
            # Ensure the difference between start and end is always 128
            for j in range(3):
                if end[j] - start[j] != 128:
                    if end[j] - start[j] < 128:
                        # If the range is less than 128, adjust the start
                        start[j] = end[j] - 128
                    else:
                        # If the range is more than 128, adjust the end
                        end[j] = start[j] + 128

            # Crop the mask to the center
            cropped_mask = labeled_mask_copy[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            cropped_mask[cropped_mask!=-1] = 1
            cropped_masks.append(cropped_mask)
            starts.append(start)
            ends.append(end)
            labeled_masks.append(labeled_mask_copy)

        return cropped_masks, starts, ends, labeled_masks

    def load_data(self, load_dir):
            img = nib.load(load_dir)
            data = img.get_fdata()

            data = data.astype(np.float32)

            if data.ndim == 2:
                data = np.expand_dims(data, axis=0)
                data = np.expand_dims(data, axis=1)

            return torch.as_tensor(data, dtype=torch.float32)
