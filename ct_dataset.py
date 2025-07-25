import os
import torch
import random
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from ct_visualizer import CTVisualizer
from ct_masking import CTMask
from ct_config import debug

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from PIL import Image


class CTDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path # Ignore and use dataset folder under job_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.patient_ids = None
        self.X_train = None
        self.Y_train = None
        self.X_val = None
        self.Y_val = None
        self.X_test = None
        self.Y_test = None
        self.X_train_masked = []
        self.X_val_masked = []
        self.X_test_masked = []
        self.test_idx = []

        # number_of_ct_patients = 5
        # self.images, self.labels, self.patient_ids = self.load_dataset(f"datasets/liver_dataset_{number_of_ct_patients}.npz")
        self.images, self.labels, self.patient_ids = self.load_dataset(self.path)

        if debug:
            print(self.patient_ids)
            print('Len (X, Y, Patients):', len(self.images), len(self.labels), len(self.patient_ids))
            print(f'Sample Patient Shapes ({self.patient_ids[2]}): X[2] Y[2]:', self.images[2].shape, self.labels[2].shape)

        if debug:
            self.print_samples()

        # Load all images and labels
        # for root, dirs, files in os.walk(path):
        #     for file in files:
        #         if file.endswith('.nii.gz'):
        #             self.images.append(os.path.join(root, file))
        #             # Assuming the label is stored in a separate file with the same name but different extension
        #             # You may need to modify this line based on your actual label storage
        #             self.labels.append(int(file.split('_')[0]))

    def load_dataset(self, name):

        data = np.load(name, allow_pickle=True)

        X_all = data["X_all"]

        Y_all = data["Y_all"]

        patient_ids = data["patient_ids"]

        return X_all, Y_all, patient_ids

    def print_summary(self):
        print("###### Summary of the dataset")
        print("Number of patients: ", len(self.patient_ids))
        print(f"Train X shape: {self.X_train.shape} Train Masked: {self.X_train_masked.shape}")
        print(f"Train Y shape: {self.Y_train.shape}")
        print(f"Val X shape: {self.X_val.shape} Val X Masked: {self.X_val_masked.shape}")
        print(f"Val Y shape: {self.Y_val.shape}")
        print(f"Test X shape: {self.X_test.shape} Test X Masked: {self.X_test_masked.shape}")
        print(f"Test Y shape: {self.Y_test.shape}")
        print("###### End Summary ##########")


    def print_samples(self):
        ctVisualizer = CTVisualizer()

        # Show Samples
        number_of_samples = 2
        number_of_slices = 4
        for i in range(number_of_samples):
            print(f"Sample {i}: liver_{i}.nii.gz")
            ctVisualizer.display_XY_samples_v2(self.images[i], self.labels[i], max_slices=number_of_slices)

    def print_XY_samples(self):
        self.print_masked_samples(self.X_train, self.X_train_masked, "Train")
        self.print_masked_samples(self.X_val, self.X_val_masked, "Validation")
        self.print_masked_samples(self.X_test, self.X_test_masked, "Test")


    def print_masked_samples(self, orig, masked, title):
        number_of_samples = 4

        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
        # Flatten the axes array for easy iteration
        axes = axes.flatten()
        i = 0

        for i in range(number_of_samples):
            img = orig[[i], :, :, :]
            img = img.squeeze()
            mask = masked[[i], :, :]
            mask = mask.squeeze()

            ax = axes[i*2]
            ax.imshow(img, cmap='gray')
            ax.set_facecolor('black')
            ax.title.set_text(f"Image: {i}")

            ax = axes[(i*2)+1]
            ax.imshow(mask, cmap='gray')
            ax.set_facecolor('black')
            ax.title.set_text(f"Mask: {i}")
            i += 1

        fig.suptitle(title)
        plt.show()

        # plt.figure(figsize=(6, 3 * number_of_samples))

        # for i in range(number_of_samples):
        #     print(f"Sample Masked and Unmasked Train and Val")
        #     img = self.X_train[[i], :, :, :]
        #     img = img.squeeze()
        #     mask = self.X_train_masked[[i], :, :]
        #     mask = mask.squeeze()
        #
        #     plt.subplot(num_samples, 2, 2 * i + 1)
        #     plt.imshow(img, cmap='gray')
        #     plt.title(f"X[{i}] CT")
        #     plt.axis('off')
        #
        #     plt.subplot(num_samples, 2, 2 * i + 2)
        #     plt.imshow(mask, cmap='gray')
        #     plt.title(f"X_Masked[{i}] CT")
        #     plt.axis('off')
        #
        # plt.tight_layout()
        # plt.show()



    def split_train_test(self, number_of_ct_patients):
        # Split patients into train, validation, test (by index)
        # Create list of slices
        train_idx, test_idx = train_test_split(range(number_of_ct_patients), test_size=0.2, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

        # print("number_fine_tuned_patients: ", number_fine_tuned_patients)
        print("train idx: ", train_idx, "Length: ", len(train_idx), type(train_idx))
        print("test idx: ", test_idx, "Length: ", len(test_idx), type(test_idx))
        print("val idx: ", val_idx, "Length: ", len(val_idx), type(val_idx))
        # Combine slices from selected patients
        self.X_train = np.concatenate([self.images[i] for i in train_idx])
        self.Y_train = np.concatenate([self.labels[i] for i in train_idx])

        self.X_val = np.concatenate([self.images[i] for i in val_idx])
        self.Y_val = np.concatenate([self.labels[i] for i in val_idx])

        self.X_test = np.concatenate([self.images[i] for i in test_idx])
        self.Y_test = np.concatenate([self.labels[i] for i in test_idx])
        self.test_idx = test_idx
        self.val_idx = val_idx

        print("Splits by patients:")

        print(len(train_idx), len(val_idx), len(test_idx))

        print("Train Shape: " ,self.X_train.shape, "Validation Shape: ", self.X_val.shape, "Test Shape: ", self.X_test.shape)

        # Test image
        # image_0 = X_train[[0], :, print_smaples:, :]
        # ct = X[idx, ..., 0]

        # print("Image 0: ", image_0.shape)
        # plt.imshow(image_0.squeeze(), cmap='gray')
        # plt.show()

    def split_finetune(self, number_of_ct_patients):
        # if True: # Remove and stay with else only
        #     train_idx, test_idx = train_test_split(range(number_of_ct_patients), test_size=0.1, random_state=42)
        # else:
        #     train_idx, test_idx = train_test_split(range(number_of_ct_patients), test_size=0.2, random_state=42)

        train_idx, test_idx = train_test_split(range(number_of_ct_patients), test_size=0.2, random_state=42)
        train_idx = test_idx

        # train_idx = [55, 40, 19, 31, 115, 56, 69, 105, 81, 26, 95, 27, 64, 4, 97, 100, 36, 80, 93, 84, 18, 10, 122, 11,
        #              127, 45, 70]
        # Shuffle the list to ensure randomness
        random.shuffle(train_idx)
        # Calculate the split index
        split_index = int(len(train_idx) * 0.8)
        # Split the list into two parts
        test_idx = train_idx[split_index:]  # 20% of the data
        train_idx = train_idx[:split_index]  # 80% of the data

        random.shuffle(train_idx)
        split_index = int(len(train_idx) * 0.8)
        val_idx = train_idx[split_index:]
        train_idx = train_idx[:split_index]  # 80% of the data

        # print("number_fine_tuned_patients: ", number_fine_tuned_patients)
        print("train idx: ", train_idx, "Length: ", len(train_idx), type(train_idx))
        print("test idx: ", test_idx, "Length: ", len(test_idx), type(test_idx))
        print("val idx: ", val_idx, "Length: ", len(val_idx), type(val_idx))
        # Combine slices from selected patients
        self.X_train = np.concatenate([self.images[i] for i in train_idx])
        self.Y_train = np.concatenate([self.labels[i] for i in train_idx])

        self.X_val = np.concatenate([self.images[i] for i in val_idx])
        self.Y_val = np.concatenate([self.labels[i] for i in val_idx])

        self.X_test = np.concatenate([self.images[i] for i in test_idx])
        self.Y_test = np.concatenate([self.labels[i] for i in test_idx])
        self.test_idx = test_idx
        self.val_idx = val_idx

        print("###### Summary of the dataset")
        print("Number of patients: ", len(self.patient_ids))
        print(f"Train X shape: {self.X_train.shape}")
        print(f"Train Y shape: {self.Y_train.shape}")
        print(f"Val X shape: {self.X_val.shape}")
        print(f"Val Y shape: {self.Y_val.shape}")
        print(f"Test X shape: {self.X_test.shape}")
        print(f"Test Y shape: {self.Y_test.shape}")
        print("###### End Summary ##########")

    def split_supervised(self, number_of_ct_patients):
        # Split patients into train, val, test (by index)
        train_idx, test_idx = train_test_split(range(number_of_ct_patients), test_size=0.2, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

        # Combine slices from selected patients
        self.X_train = np.concatenate([self.images[i] for i in train_idx])
        self.Y_train = np.concatenate([self.labels[i] for i in train_idx])

        self.X_val = np.concatenate([self.images[i] for i in val_idx])
        self.Y_val = np.concatenate([self.labels[i] for i in val_idx])

        self.X_test = np.concatenate([self.images[i] for i in test_idx])
        self.Y_test = np.concatenate([self.labels[i] for i in test_idx])

        self.test_idx = test_idx
        self.val_idx = val_idx

        print("###### Summary of the dataset")
        print("Number of patients: ", len(self.patient_ids))
        print(f"Train X shape: {self.X_train.shape}")
        print(f"Train Y shape: {self.Y_train.shape}")
        print(f"Val X shape: {self.X_val.shape}")
        print(f"Val Y shape: {self.Y_val.shape}")
        print(f"Test X shape: {self.X_test.shape}")
        print(f"Test Y shape: {self.Y_test.shape}")
        print("###### End Summary ##########")


    def test_transform(self):
        index = 0
        # (256, 256)
        image = self.X_train[index, ..., 0]

        # For debug purposes, print current image
        plt.imshow(image, cmap='gray')
        # plt.imshow(image.squeeze(), cmap='gray')
        plt.show()

        # Convert the ndarray to a PIL Image
        image_uint8 = (image * 255).astype(np.uint8)
        plt.imshow(image_uint8, cmap='gray')
        plt.show()

        pil_image = Image.fromarray(image_uint8, mode='L')
        plt.imshow(pil_image, cmap='gray')
        plt.show()

        # convert back to ndarray
        image_from_pil = np.array(pil_image)
        plt.imshow(image_from_pil, cmap='gray')
        plt.show()

    def mask_and_save(self):
        masker = CTMask()
        for idx, img in enumerate(self.X_train):
            img = img.squeeze()
            masked_img = masker.mask(img)
            self.X_train_masked.append(masked_img)
            if idx % 500 == 0:
                print(f'Train Processed {idx}/{len(self.X_train)}')
        print(f'Completed Train Processed {idx+1}/{len(self.X_train)}')
        self.X_train_masked = np.stack(self.X_train_masked, axis=0)

        for idx, img in enumerate(self.X_val):
            img = img.squeeze()
            masked_img = masker.mask(img)
            self.X_val_masked.append(masked_img)
            if idx % 500 == 0:
                print(f'Val Processed {idx}/{len(self.X_val)}')
        print(f'Completed Val Processed {idx+1}/{len(self.X_val)}')
        self.X_val_masked = np.stack(self.X_val_masked, axis=0)

        for idx, img in enumerate(self.X_test):
            img = img.squeeze()
            masked_img = masker.mask(img)
            self.X_test_masked.append(masked_img)
            if idx % 500 == 0:
                print(f'Test Processed {idx}/{len(self.X_test)}')
        print(f'Completed Test Processed {idx+1}/{len(self.X_test)}')
        self.X_test_masked = np.stack(self.X_test_masked, axis=0)

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, index):
        # (256, 256)
        # global image
        image_slice = self.X_train[index, ..., 0]

        # During pre-train, the label is the original image
        # label = self.Y_train[index, ..., 0]
        label_slice = self.X_train[index, ..., 0]

        # (256, 256, 1)
        # image = self.X_train[index]
        # label = self.Y_train[index]

        # For debug purposes, print current image
        plt.imshow(image_slice, cmap='gray')
        # plt.imshow(image.squeeze(), cmap='gray')
        plt.show()

        # Convert the ndarray to a PIL Image
        img = Image.fromarray(image_slice, mode='L')
        label = Image.fromarray(label_slice, mode='L')

        # For debug, print image after conversion to image
        # For debug purposes, print current image
        plt.imshow(img, cmap='gray')
        # plt.imshow(image.squeeze(), cmap='gray')
        plt.show()

        ###############
        # using uint8 as type - suggested for Image conversion
        image_uint8 = (image_slice * 255).astype(np.uint8)
        plt.imshow(image_uint8, cmap='gray')
        plt.show()

        # convert and show 2d - testing view
        pil_image = Image.fromarray(image_uint8, mode='L')
        plt.imshow(pil_image, cmap='gray')
        plt.show()

        # convert back to ndarray
        image_from_pil = np.array(pil_image)
        plt.imshow(image_from_pil, cmap='gray')
        plt.show()
        #################

        # Apply any necessary preprocessing (e.g., normalization)
        if self.transform:
            image = self.transform(img)

        test_image = image.numpy().squeeze()
        # For debug purposes, print current image
        plt.imshow(test_image, cmap='gray')
        # plt.imshow(image.squeeze(), cmap='gray')
        plt.show()

        # Convert the image to a PyTorch tensor
        # image = torch.from_numpy(image).unsqueeze(0)  # Add a channel dimension

        return image, label

# Example usage:
# transform = lambda x: (x - np.mean(x)) / np.std(x)  # Simple normalization transform
# dataset = CTDataSet('path/to/ct_data/train', transform=transform)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)