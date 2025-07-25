from functools import partial
import matplotlib.pyplot as plt
import torch

import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

import numpy as np
from ct_config import debug

class CTMask():

    def __init__(self, img_size=256, patch_size=16):

        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self. num_patches = (img_size // patch_size) ** 2  # 256 patches for 256x256 with 16x16 patches
        self.mask_ratio = 0.75
        self.num_mask = int(self.num_patches * self.mask_ratio)

    def mask(self, img):
        # image_slice = imgs[index, ..., 0]
        image_slice = img
        image_tensor = torch.tensor(image_slice)

        if debug:
            plt.imshow(image_slice, cmap='gray')
            plt.title("Slice 0 of Train data _before_ masking")
            plt.show()

            plt.imshow(image_tensor, cmap='gray')
            plt.title("Slice 0 of Train data _before_ masking as tensor")
            plt.show()


        # Step 1: Divide into patches
        patches = self.img_to_patches(image_tensor)

        # Step 2: Generate mask
        num_total = patches.shape[0]
        perm = torch.randperm(num_total)
        masked_indices = perm[:self.num_mask]
        visible_indices = perm[self.num_mask:]

        # Step 3: Apply mask
        visible_patches = patches[visible_indices]  # input to encoder
        mask = torch.ones(num_total)
        mask[visible_indices] = 0  # 1 = masked, 0 = visible

        # For reconstruction, you can send mask + latent tokens to the decoder
        reconstructed_img = self.reconstruct_image_from_patches(visible_patches, visible_indices, image_tensor.shape)

        if debug:
            plt.imshow(reconstructed_img, cmap='gray')
            plt.title("Reconstructed Image after masking")
            plt.show()

            # Optional debug
            print(f"Original patches: {patches.shape}")
            print(f"Visible patches: {visible_patches.shape}")
            print(f"Mask shape: {mask.shape}, masked: {int(mask.sum().item())} patches")

        return reconstructed_img



    def img_to_patches(self, img):
        # C, H, W = img.shape
        # assert H % self.patch_size == 0 and W % self.patch_size == 0
        # patches = img.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        # patches = patches.permute(1, 2, 0, 3, 4)  # [num_h, num_w, C, pH, pW]
        # patches = patches.reshape(-1, C * self.patch_size * self.patch_size)  # [num_patches, patch_dim]
        # return patches

        H, W = img.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        # Unfold along both H (dim 0) and W (dim 1)
        patches = img.unfold(0, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size)
        # patches shape: [num_h, num_w, pH, pW]
        patches = patches.reshape(-1, self.patch_size * self.patch_size)  # [num_patches, patch_dim]
        return patches

    def reconstruct_image_from_patches(self, patches, visible_indices, original_shape):
        H, W = original_shape
        patch_size = self.patch_size
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        # Create empty image tensor
        reconstructed = torch.zeros(H, W)
        # Place visible patches back into the image
        for i, idx in enumerate(visible_indices):
            row = idx // num_patches_w
            col = idx % num_patches_w
            patch = patches[i].reshape(patch_size, patch_size)
            reconstructed[
            row * patch_size: (row + 1) * patch_size,
            col * patch_size: (col + 1) * patch_size,
            ] = patch
        return reconstructed




