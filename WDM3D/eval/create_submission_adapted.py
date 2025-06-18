import os

import nibabel as nib
import numpy as np
import torch as th


def adapt(input_data=None, samples_dir=None, adapted_samples_dir=None):

    for root, dirs, files in os.walk(samples_dir):
        for file_id in files:

            if file_id.endswith("-inference.nii.gz"):
                folder_name = file_id.split("-")[2] + "-" + file_id.split("-")[3]

                nifti_file = nib.load(str(root) + "/" + str(file_id))

                org_nifti = nib.load(
                    str(input_data)
                    + "/BraTS-GLI-"
                    + str(folder_name)
                    + "/BraTS-GLI-"
                    + str(folder_name)
                    + "-t1n-voided.nii.gz"
                )
                target_header = org_nifti.header

                healthy_mask = nib.load(
                    str(input_data)
                    + "/BraTS-GLI-"
                    + str(folder_name)
                    + "/BraTS-GLI-"
                    + str(folder_name)
                    + "-mask.nii.gz"
                )

                np_org = np.asarray(org_nifti.dataobj)
                original_dtype = org_nifti.get_data_dtype()

                np_healthy_mask = np.asarray(healthy_mask.dataobj)
                np_org[np_healthy_mask == 1] = 0

                np_org_clipped = np.percentile(np_org, [0.5, 99.5])

                np_redef = np.asarray(nifti_file.dataobj)

                np_redef = (np_redef - np.min(np_redef)) / (np.max(np_redef) - np.min(np_redef))

                clipped_image = np.clip(np_redef, 0, 1)
                start = np.min(np_org_clipped)
                end = np.max(np_org_clipped)
                width = end - start
                norm_img = (clipped_image - clipped_image.min()) / (
                        clipped_image.max() - clipped_image.min()
                ) * width + start

                cropped_image = norm_img[8:-8, 8:-8, 50:-51]
                cropped_image = cropped_image.astype(original_dtype)

                nib.save(
                    nib.Nifti1Image(cropped_image, None, target_header),
                    adapted_samples_dir + "/" + str(file_id),
                )

        break