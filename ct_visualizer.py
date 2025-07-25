import matplotlib.pyplot as plt
import numpy as np
import math

class CTVisualizer:
    def __init__(self, hu_window=(30, 180)):
        self.hu_window = hu_window

    def display_ct_data(self, img, slice_num=-1):
        # Extract dimensions
        sagittal_slices = img.shape[0]  # x-axis
        coronal_slices = img.shape[1]  # y-axis
        axial_slices = img.shape[2]  # z-axis

        # If no specific slice is requested, use the middle
        if slice_num == -1:
            first = sagittal_slices // 2
            second = coronal_slices // 2
            last = axial_slices // 2
        else:
            first = second = last = slice_num

        # Extract slices
        sagittal = img[first, :, :]
        coronal = img[:, second, :]
        axial = img[:, :, last]

        print_info = False
        if print_info:
            # Formatted Output
            print("\nAvailable Planes: Sagittal | Coronal | Axial\n")

            print("Sagittal Plane")
            print(f"    Shape: {sagittal.shape}")
            print(f"    Slices: {sagittal_slices} (along x-axis)")
            print("    Description: Divides the body into left and right sections.\n")

            print("Coronal Plane")
            print(f"    Shape: {coronal.shape}")
            print(f"    Slices: {coronal_slices} (along y-axis)")
            print("    Description: Divides the body into front (anterior) and back (posterior) sections.\n")

            print("Axial (Transverse) Plane")
            print(f"    Shape: {axial.shape}")
            print(f"    Slices: {axial_slices} (along z-axis)")
            print("    Description: Divides the body into upper (superior) and lower (inferior) sections.\n")

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))

        axes[0].imshow(sagittal, cmap='gray')
        axes[0].set_title(f'Sagittal View [{first}, :, :]')

        axes[1].imshow(coronal, cmap='gray')
        axes[1].set_title(f'Coronal View [:, {second}, :]')

        axes[2].imshow(axial, cmap='gray')
        axes[2].set_title(f'Axial (Transverse) View [:, :, {last}]')

        plt.tight_layout()
        plt.show()

    def display_slices(self, ct_slices, label_slices, mode='overlay', max_slices=5):
        assert mode in ['overlay', 'split'], "mode must be 'overlay' or 'split'"

        num_slices = len(ct_slices)
        if max_slices != -1:
            num_slices = min(num_slices, max_slices)

        fig, axes = plt.subplots(num_slices, 2, figsize=(5, 3 * num_slices),
                                 gridspec_kw={'wspace': 0.05, 'hspace': 0.2})

        if num_slices == 1:
            axes = np.expand_dims(axes, axis=0)

        for idx in range(num_slices):
            ct = ct_slices[idx]
            label = label_slices[idx]

            axes[idx, 0].imshow(ct, cmap='gray')
            axes[idx, 0].set_title(f"CT Slice {idx}")
            axes[idx, 0].axis('off')

            if mode == 'overlay':
                axes[idx, 1].imshow(ct, cmap='gray')
                axes[idx, 1].imshow(label, cmap='Reds', alpha=0.4)
                axes[idx, 1].set_title("Overlay")
            else:  # mode == 'split'
                axes[idx, 1].imshow(label, cmap='gray')
                axes[idx, 1].set_title("Label Mask")

            axes[idx, 1].axis('off')

        plt.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.02, hspace=0.15, wspace=0.05)
        plt.show()

    def display_single_slice(self, ct_slices, label_slices, slice_index=0):

        fig, axes = plt.subplots(1, 3, figsize=(10, 3), gridspec_kw={'wspace': 0.05, 'hspace': 0.2})

        axes = np.expand_dims(axes, axis=0)

        idx = 0
        ct = ct_slices[slice_index]
        label = label_slices[slice_index]

        axes[idx, 0].imshow(ct, cmap='gray')
        axes[idx, 0].set_title(f"CT Slice {slice_index}")
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(label, cmap='gray')
        axes[idx, 1].set_title("Label Mask")
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(ct, cmap='gray')
        axes[idx, 2].imshow(label, cmap='Reds', alpha=0.4)
        axes[idx, 2].set_title("Overlay")
        axes[idx, 2].axis('off')

        plt.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.02, hspace=0.15, wspace=0.05)
        plt.show()

    def display_XY_samples(self, X, Y, max_slices=5, binary=True):
        num_samples = min(len(X), max_slices)
        plt.figure(figsize=(6, 3 * num_samples))

        for idx in range(num_samples):
            ct = X[idx, ..., 0]

            if binary:
                mask = Y[idx, ..., 0]
            else:
                mask = Y[idx]

            plt.subplot(num_samples, 2, 2 * idx + 1)
            plt.imshow(ct, cmap='gray')
            plt.title(f"X[{idx}] CT")
            plt.axis('off')

            plt.subplot(num_samples, 2, 2 * idx + 2)
            if binary:
                plt.imshow(mask, cmap='gray')
            else:
                plt.imshow(mask, cmap='tab10', vmin=0, vmax=np.max(mask))
            plt.title(f"Y[{idx}] {'Binary' if binary else 'Multi-class'} Mask")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    import matplotlib.pyplot as plt
    import numpy as np

    def display_XY_samples_v2(self, X, Y, max_slices=5, binary=True):
        num_samples = min(len(X), max_slices)
        num_rows = (num_samples + 1) // 2  # 2 pairs per row

        plt.figure(figsize=(12, 3 * num_rows))

        for idx in range(num_samples):
            ct = X[idx, ..., 0]
            mask = Y[idx, ..., 0] if binary else Y[idx]

            # Row and column positions
            row = idx // 2
            col_offset = (idx % 2) * 2  # 0 or 2 (CT, Mask)

            # Subplot index calculation
            subplot_idx_ct = row * 4 + col_offset + 1
            subplot_idx_mask = row * 4 + col_offset + 2

            # Plot CT
            plt.subplot(num_rows, 4, subplot_idx_ct)
            plt.imshow(ct, cmap='gray')
            plt.title(f"X[{idx}] CT")
            plt.axis('off')

            # Plot Mask
            plt.subplot(num_rows, 4, subplot_idx_mask)
            if binary:
                plt.imshow(mask, cmap='gray')
            else:
                plt.imshow(mask, cmap='tab10', vmin=0, vmax=np.max(mask))
            plt.title(f"Y[{idx}] {'Binary' if binary else 'Multi-class'} Mask")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def save_slices_panel(self, ct_slices, label_slices, resize_fn, max_slices=-1, target_path="output.png"):
        if max_slices > 0:
            ct_slices = ct_slices[:max_slices]
            label_slices = label_slices[:max_slices]

        num_slices = len(ct_slices)
        cols = 6
        rows = math.ceil(num_slices / cols)
        fig, axes = plt.subplots(2 * rows, cols, figsize=(2 * cols, 4 * rows))

        for idx in range(num_slices):
            ct, label = resize_fn(ct_slices[idx], label_slices[idx], hu_window=self.hu_window)
            row_img = (idx // cols) * 2
            col = idx % cols

            axes[row_img, col].imshow(ct, cmap='gray')
            axes[row_img, col].axis('off')
            axes[row_img, col].set_title(f"Slice {idx}", fontsize=8)

            axes[row_img + 1, col].imshow(ct, cmap='gray')
            axes[row_img + 1, col].imshow(label, cmap='Reds', alpha=0.4)
            axes[row_img + 1, col].axis('off')

        for i in range(num_slices, rows * cols):
            axes[(i // cols) * 2, i % cols].axis('off')
            axes[(i // cols) * 2 + 1, i % cols].axis('off')

        plt.tight_layout()
        plt.savefig(target_path, dpi=200)
        plt.close()
