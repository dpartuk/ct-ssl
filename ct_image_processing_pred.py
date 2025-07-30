import nibabel as nib
import numpy as np
import cv2
import os
from tabulate import tabulate
from skimage.transform import resize  # assumes scikit-image is available


class ImageProcessing:
    def __init__(self):
        pass


    def print_plane_info(self, info):
        print("CT Plane Summary Table:")

        table = []
        headers = ["Plane", "Axis", "Slices", "Labeled Slices", "Description"]

        for plane, details in info['planes'].items():
            table.append([
                plane,
                details.get('axis', '-'),
                details.get('slices', '-'),
                details.get('labeled_slices', '-'),
                details.get('description', '-')
            ])

        print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

    def print_ct_info(self, info):

        print('------------------------------------------------------------------------')
        print(f"CT Summary for {info['name']} ({info['kind']})")
        print('------------------------------------------------------------------------')

        print(f"Overall Shape: {info['shape']} (x, y, z)")

        self.print_plane_info(info)

        # print("Available Planes:\n")
        #
        # for plane, details in info['planes'].items():
        #     print(f"{plane} Plane")
        #     print(f"    Axis: {details['axis']}")
        #     print(f"    Slices: {details['slices']}")
        #     if details.get('description'):
        #         print(f"    Description: {details['description']}\n")
        #     if details.get('labeled_slices'):
        #         print(f"    Labeled Slices: {details['labeled_slices']}\n")


    def loadCTImage(self, imagePath):
        import nibabel as nib

        ct_image = nib.load(imagePath).get_fdata()

        # Dimensions
        shape = ct_image.shape
        sagittal_slices = shape[0]  # x-axis
        coronal_slices = shape[1]  # y-axis
        axial_slices = shape[2]  # z-axis

        info = {
            'name': os.path.basename(imagePath),
            'shape': shape,
            'kind': 'Image',
            'planes': {
                'Sagittal': {
                    'axis': 'x',
                    'slices': sagittal_slices,
                    'description': 'Divides the body into left and right sections.'
                },
                'Coronal': {
                    'axis': 'y',
                    'slices': coronal_slices,
                    'description': 'Divides the body into front (anterior) and back (posterior) sections.'
                },
                'Axial': {
                    'axis': 'z',
                    'slices': axial_slices,
                    'description': 'Divides the body into upper (superior) and lower (inferior) sections.'
                }
            }
        }

        return ct_image, info

    def loadCTLabel(self, labelPath):

        ct_label = nib.load(labelPath).get_fdata()
        shape = ct_label.shape

        # Labeled slices per plane
        sagittal_labeled = [i for i in range(shape[0]) if np.max(ct_label[i, :, :]) > 0]
        coronal_labeled = [i for i in range(shape[1]) if np.max(ct_label[:, i, :]) > 0]
        axial_labeled = [i for i in range(shape[2]) if np.max(ct_label[:, :, i]) > 0]

        info = {
            'name': os.path.basename(labelPath),
            'shape': shape,
            'kind': 'Label',
            'planes': {
                'Sagittal': {
                    'axis': 'x',
                    'slices': shape[0],
                    'labeled_slices': len(sagittal_labeled),
                },
                'Coronal': {
                    'axis': 'y',
                    'slices': shape[1],
                    'labeled_slices': len(coronal_labeled),
                },
                'Axial': {
                    'axis': 'z',
                    'slices': shape[2],
                    'labeled_slices': len(axial_labeled),
                }
            }
        }

        return ct_label, info#, axial_labeled

    # Function to load 3D CT and label volumes, and extract relevant 2D slices
    def load_image_and_labels_slices(self, ct_vol_path,
                                     label_vol_path,
                                     only_labeled_slices=True,
                                     plane='Axial'):
        """
        Load CT and label volumes and return 2D slices along the specified plane.

        Parameters:
        - ct_vol_path: path to CT volume (.nii.gz)
        - label_vol_path: path to label volume (.nii.gz)
        - only_labeled_slices: if True, return only slices that contain labels
        - plane: one of 'Axial', 'Coronal', 'Sagittal'

        Returns:
        - ct_slices: list of 2D CT slices
        - label_slices: list of 2D label slices
        - info: metadata including plane and slice counts
        """

        assert plane in ['Axial', 'Coronal', 'Sagittal'], "mode must be 'Axial', 'Coronal', 'Sagittal'"

        ct_img = nib.load(ct_vol_path).get_fdata()
        label_img = nib.load(label_vol_path).get_fdata()

        shape = 0
        if plane == 'Axial':
            axis = 2
            slice_func = lambda img, i: img[:, :, i]
            plane_desc = 'Transverse (Axial) [:, :, i]'
            shape = ct_img[:, :, 0].shape
        elif plane == 'Coronal':
            axis = 1
            slice_func = lambda img, i: img[:, i, :]
            plane_desc = 'Coronal [:, i, :]'
            shape = ct_img[:, 1, :].shape
        elif plane == 'Sagittal':
            axis = 0
            slice_func = lambda img, i: img[i, :, :]
            plane_desc = 'Sagittal [i, :, :]'
            shape = ct_img[1, :, :].shape
        else:
            raise ValueError("Invalid plane. Choose from 'Axial', 'Coronal', 'Sagittal'.")

        print(ct_img.shape)
        total_slices = ct_img.shape[axis]
        total_labeled_slices = sum(np.max(slice_func(label_img, i)) > 0 for i in range(total_slices))

        if only_labeled_slices:
            slices = [i for i in range(total_slices) if np.max(slice_func(label_img, i)) > 0]
        else:
            slices = range(total_slices)

        ct_slices = [slice_func(ct_img, i) for i in slices]
        label_slices = [slice_func(label_img, i) for i in slices]

        info = {
            'total_slices': total_slices,
            'total_labeled_slices': total_labeled_slices,
            'Plane': plane_desc,
            'Shape': shape,
        }

        return ct_slices, label_slices, info

    # Function to resize a single CT and label slice
    def resize_ct_and_label(self, ct_slice,
                            label_slice,
                            target_size=(256, 256),
                            hu_window=(30, 180),
                            binary=True):

        level, width = hu_window
        min_hu = level - width // 2
        max_hu = level + width // 2

        ct_slice = np.clip(ct_slice, min_hu, max_hu)
        ct_slice = (ct_slice - min_hu) / (max_hu - min_hu)

        ct_resized = cv2.resize(ct_slice, target_size, interpolation=cv2.INTER_LINEAR)

        if binary:
            label_resized = cv2.resize((label_slice > 0).astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
        else:
            label_resized = cv2.resize(label_slice.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)

        return ct_resized, label_resized

    # Function to resize all slices in a volume
    def resize_all_slices(self, ct_slices,
                          label_slices,
                          target_size=(256, 256),
                          hu_window=(30, 180),
                          binary=True):

        resized_ct_slices = []
        resized_label_slices = []

        for ct_slice, label_slice in zip(ct_slices, label_slices):

            ct_resized, label_resized = self.resize_ct_and_label(
                                                        ct_slice,
                                                        label_slice,
                                                        target_size=target_size,
                                                        hu_window=hu_window,
                                                        binary=binary)

            resized_ct_slices.append(ct_resized)

            resized_label_slices.append(label_resized)

        return resized_ct_slices, resized_label_slices

    # Final wrapper to prepare X (images) and Y (labels) for training
    def prepare_segmentation_dataset(self, ct_path,
                                     label_path,
                                     target_size=(256, 256),
                                     hu_window=(30, 180),
                                     binary=True,
                                     labeled_only=True):

        ct_slices, label_slices, info = self.load_image_and_labels_slices(ct_path, label_path, only_labeled_slices=labeled_only)

        resized_ct_slices, resized_label_slices = self.resize_all_slices(
            ct_slices, label_slices, target_size, hu_window, binary=binary
        )

        X = np.array([img[..., np.newaxis].astype(np.float32) for img in resized_ct_slices])
        if binary:
            Y = np.array([mask[..., np.newaxis].astype(np.uint8) for mask in resized_label_slices])
        else:
            Y = np.array([mask.astype(np.uint8) for mask in resized_label_slices])  # shape: (H, W)

        return X, Y, info  # X: float32, Y: uint8 (binary or multi-class depending on flag)

    def create_dataset(self, ct_folder_path,
                       label_folder_path,
                       target_size=(256, 256),
                       hu_window=(30, 180),
                       binary=True,
                       number_of_ct_patients=10,
                       labeled_only=True,
                       patient_offset=0):
        X_all = []
        Y_all = []
        patient_ids = []
        total_labeled_slices = 0
        total_slices = 0
        for i in range(number_of_ct_patients):
            image_name = f"liver_{i+patient_offset}.nii.gz"
            ct_vol = f"{ct_folder_path}/{image_name}"
            label_vol = f"{label_folder_path}/{image_name}"

            X, Y, info = self.prepare_segmentation_dataset(ct_vol,
                                                           label_vol,
                                                           binary=binary,
                                                           target_size=target_size,
                                                           hu_window=hu_window,
                                                           labeled_only=labeled_only)

            print(f"liver_{i} slices: {info['total_labeled_slices']} / {info['total_slices']}")

            total_labeled_slices += info['total_labeled_slices']
            total_slices += info['total_slices']

            X_all.append(X)  # list of arrays per patient
            Y_all.append(Y)

            patient_ids.append(f"liver_{i}")  # optional, for metadata tracking

        return X_all, Y_all, patient_ids, {'total_labeled_slices': total_labeled_slices, 'total_slices': total_slices}

    import nibabel as nib
    import numpy as np
    import cv2

    def create_test_dataset(self,
                            ct_folder_path,
                            target_size=(256, 256),
                            hu_window=(30, 180),
                            number_of_ct_patients=10,
                            patient_offset=132):
        X_all_test = []
        total_slices = 0

        for i in range(number_of_ct_patients):
            image_name = f"liver_{i+ patient_offset}.nii.gz"
            ct_vol_path = f"{ct_folder_path}/{image_name}"

            # Load full CT volume
            ct_img = nib.load(ct_vol_path).get_fdata()

            # Extract axial slices (default)
            slices = [ct_img[:, :, idx] for idx in range(ct_img.shape[2])]

            # Preprocess all slices
            level, width = hu_window
            min_hu = level - width // 2
            max_hu = level + width // 2

            processed_slices = []

            for ct_slice in slices:
                # HU windowing
                ct_slice = np.clip(ct_slice, min_hu, max_hu)
                ct_slice = (ct_slice - min_hu) / (max_hu - min_hu)

                # Resize
                ct_resized = cv2.resize(ct_slice, target_size, interpolation=cv2.INTER_LINEAR)

                # Add channel dimension
                ct_resized = ct_resized[..., np.newaxis].astype(np.float32)

                processed_slices.append(ct_resized)

            # Stack slices into patient array: (num_slices, H, W, 1)
            X_patient = np.stack(processed_slices, axis=0)
            X_all_test.append(X_patient)

            print(f"liver_{i} slices: {X_patient.shape[0]}")
            total_slices += X_patient.shape[0]

        print(f"Total test slices loaded: {total_slices}")
        return X_all_test, {'total_slices': total_slices}

    def create_test_dataset_v1(self, ct_folder_path,
                            number_of_ct_patients=1,
                            patient_offset=132,
                            target_size=(256, 256)):
        X_all = []
        total_slices = 0

        for i in range(number_of_ct_patients):
            image_name = f"liver_{i + patient_offset}.nii.gz"
            ct_vol = f"{ct_folder_path}/{image_name}"

            ct_img = nib.load(ct_vol).get_fdata()

            axis = 2  # slicing direction
            slice_func = lambda img, i: img[:, :, i]
            plane_desc = 'Transverse (Axial) [:, :, i]'

            patient_slices = ct_img.shape[axis]
            slices = range(patient_slices)

            ct_slices = []

            for s in slices:
                slice_img = slice_func(ct_img, s)

                # Resize to target size (256x256)
                resized = resize(slice_img, target_size, order=1, preserve_range=True, anti_aliasing=True)

                # Add channel dimension (H, W, 1)
                resized = resized[..., np.newaxis]

                ct_slices.append(resized)

            # Convert list of slices to array: (num_slices, 256, 256, 1)
            ct_array = np.stack(ct_slices, axis=0)
            X_all.append(ct_array)

            print(f"liver_{i} slices: {patient_slices}")
            total_slices += patient_slices

        return X_all, {'total_slices': total_slices}

    def save_dataset(self, X_all,
                     Y_all,
                     patient_ids,
                     name):

        np.savez_compressed(name,
                            X_all=np.array(X_all, dtype=object),
                            Y_all=np.array(Y_all, dtype=object),
                            patient_ids=np.array(patient_ids, dtype=object))

    def load_dataset(self, name):

        data = np.load(name, allow_pickle=True)

        X_all = data["X_all"]

        Y_all = data["Y_all"]

        patient_ids = data["patient_ids"]

        return X_all, Y_all, patient_ids

    import numpy as np

    def generate_masked_images(self, X, patch_size=16, mask_ratio=0.75, seed=None):
        """
        Generate masked versions of 2D CT images by masking a patch-wise random subset.

        Args:
            X (np.ndarray): Array of shape (N, H, W), grayscale images.
            patch_size (int): Size of square patches to mask (e.g., 16).
            mask_ratio (float): Ratio of patches to mask (e.g., 0.75).
            seed (int, optional): Random seed for reproducibility.

        Returns:
            masked_X (np.ndarray): Same shape as X, with 75% patches masked to 0.
        """
        if seed is not None:
            np.random.seed(int(seed))

        N, H, W = X.shape
        assert H % patch_size == 0 and W % patch_size == 0, "Image size must be divisible by patch_size"

        masked_X = X.copy()
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        total_patches = num_patches_h * num_patches_w
        num_masked = int(total_patches * mask_ratio)

        for idx in range(N):
            patch_indices = np.arange(total_patches)
            np.random.shuffle(patch_indices)
            masked_indices = patch_indices[:num_masked]

            for patch_id in masked_indices:
                i = patch_id // num_patches_w
                j = patch_id % num_patches_w
                y_start = i * patch_size
                x_start = j * patch_size
                masked_X[idx, y_start:y_start + patch_size, x_start:x_start + patch_size] = 0  # mask to 0

        return masked_X
