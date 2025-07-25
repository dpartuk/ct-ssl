import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color.rgb_colors import white
from ipywidgets import interact, IntSlider
from mpl_toolkits.mplot3d import Axes3D

class ThreeDExploration:
    def __init__(self):
        pass

    def plot_voxel_planes_highlighted_interactive(self, volume_shape=(10, 10, 5)):
        def plot(x, y, z):
            filled = np.zeros(volume_shape, dtype=bool)
            filled[x, :, :] = True  # sagittal
            filled[:, y, :] = True  # coronal
            filled[:, :, z] = True  # axial

            highlight = np.zeros(volume_shape, dtype=bool)
            highlight[x, y, z] = True

            colors = np.empty(volume_shape, dtype=object)
            colors[x, :, :] = 'red'
            colors[:, y, :] = 'blue'
            colors[:, :, z] = 'lightgreen'
            colors[x, y, z] = 'black'

            fig = plt.figure(figsize=(12, 6))
            gs = fig.add_gridspec(1, 3, width_ratios=[0.2, 1, 1.2])

            ax = fig.add_subplot(gs[1], projection='3d')
            ax.voxels(
                highlight,
                facecolors='yellow',
                edgecolors='black',
                linewidth=1.0,
                alpha=1.0
            )
            ax.voxels(filled, facecolors=colors, edgecolors='white', linewidth=0.5, alpha=0.75)

            ax.set_xlabel('X (Sagittal)')
            ax.set_ylabel('Y (Coronal)')
            ax.set_zlabel('Z (Axial)')
            ax.set_title('3D CT Planes with Highlighted Voxel')
            ax.set_xlim(0, volume_shape[0])
            ax.set_ylim(0, volume_shape[1])
            ax.set_zlim(0, volume_shape[2])
            ax.view_init(elev=30, azim=210)

            ax2 = fig.add_subplot(gs[2])
            img_path_2 = 'medical_images/Human_anatomy_planes.jpg'
            try:
                img = mpimg.imread(img_path_2)
                ax2.imshow(img)
                ax2.axis('off')
                ax2.set_title("Anatomy Planes")
            except FileNotFoundError:
                ax2.text(0.5, 0.5, 'Image not found', ha='center', va='center')
                ax2.axis('off')

            plt.tight_layout(pad=3.0)
            plt.show()

        interact(
            plot,
            x=IntSlider(min=0, max=volume_shape[0] - 1, step=1, value=volume_shape[0] // 2, description='X', style={'handle_color': 'red'}),
            y=IntSlider(min=0, max=volume_shape[1] - 1, step=1, value=volume_shape[1] // 2, description='Y', style={'handle_color': 'blue'}),
            z=IntSlider(min=0, max=volume_shape[2] - 1, step=1, value=volume_shape[2] // 2, description='Z', style={'handle_color': 'lightgreen'}),
        )


    def plot_voxel_planes_highlighted_v2(self, volume_shape=(10, 10, 5), voxel_coords=(5, 5, 2)):
        """
        Render 3 intersecting CT planes with a prominently highlighted central voxel.
        """
        x, y, z = voxel_coords

        # Base volume for planes
        filled = np.zeros(volume_shape, dtype=bool)
        filled[x, :, :] = True  # sagittal
        filled[:, y, :] = True  # coronal
        filled[:, :, z] = True  # axial

        # Add central voxel
        highlight = np.zeros(volume_shape, dtype=bool)
        highlight[x, y, z] = True

        # Color arrays
        colors = np.empty(volume_shape, dtype=object)
        colors[x, :, :] = 'red'
        colors[:, y, :] = 'blue'
        colors[:, :, z] = 'lightgreen'
        colors[x, y, z] = 'white'  # voxel in base

        # Create figure and grid layout with a left spacer
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[0.2, 1, 1.2])  # Added left spacer

        # Left subplot (second column): 3D voxel plot
        ax = fig.add_subplot(gs[1], projection='3d')
        ax.voxels(
            highlight,
            facecolors='yellow',
            edgecolors='black',
            linewidth=1.0,  # was 2.5 — increase it
            alpha=1.0
        )
        ax.voxels(filled, facecolors=colors, edgecolors=white, linewidth=0.5, alpha=0.75)
        # ax.voxels(highlight, facecolors='red', edgecolors='black', linewidth=2.5, alpha=1.0)

        ax.set_xlabel('X (Sagittal)')
        ax.set_ylabel('Y (Coronal)')
        ax.set_zlabel('Z (Axial)')
        ax.set_title('3D CT Planes with Highlighted Voxel')
        ax.set_xlim(0, volume_shape[0])
        ax.set_ylim(0, volume_shape[1])
        ax.set_zlim(0, volume_shape[2])
        ax.view_init(elev=30, azim=210)

        # Right subplot (last column): anatomy image
        ax2 = fig.add_subplot(gs[2])
        img_path_2 = 'medical_images/Human_anatomy_planes.jpg'
        img = mpimg.imread(img_path_2)
        ax2.imshow(img)
        ax2.axis('off')
        ax2.set_title("Anatomy Planes")

        # plt.subplots_adjust(left=0.9, right=0.95, wspace=0.4)
        plt.tight_layout(pad=3.0)
        plt.show()

    def plot_voxel_planes_highlighted(self, volume_shape=(10, 10, 5), voxel_coords=(5, 5, 2)):
        """
        Render 3 intersecting CT planes with a prominently highlighted central voxel.
        """
        x, y, z = voxel_coords

        # Base volume for planes
        filled = np.zeros(volume_shape, dtype=bool)
        filled[x, :, :] = True     # sagittal
        filled[:, y, :] = True     # coronal
        filled[:, :, z] = True     # axial

        # Add central voxel
        highlight = np.zeros(volume_shape, dtype=bool)
        highlight[x, y, z] = True

        # Color arrays
        colors = np.empty(volume_shape, dtype=object)
        colors[x, :, :] = 'lightgreen'
        colors[:, y, :] = 'lightsalmon'
        colors[:, :, z] = 'lightgray'
        colors[x, y, z] = 'red'  # voxel in base

        # Plot planes
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))

        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(filled, facecolors=colors, edgecolors=None, linewidth=0, alpha=0.85)

        # Overlay: make the voxel highly visible with bold edge
        ax.voxels(highlight, facecolors='red', edgecolors='black', linewidth=2.5, alpha=1.0)

        ax.set_xlabel('X (Sagittal)')
        ax.set_ylabel('Y (Coronal)')
        ax.set_zlabel('Z (Axial)')
        ax.set_title('3D CT Planes with Highlighted Voxel')

        ax.set_xlim(0, volume_shape[0])
        ax.set_ylim(0, volume_shape[1])
        ax.set_zlim(0, volume_shape[2])
        ax.view_init(elev=30, azim=210)

        axs[0].axis('off')

        # Right subplot: 2D image
        img_path_2 = 'medical_images/Human_anatomy_planes.jpg'
        img = mpimg.imread(img_path_2)
        axs[1].imshow(img)
        axs[1].axis('off')
        axs[1].set_title("Anatomy Planes")

        plt.tight_layout()
        plt.show()

    def generate_realistic_ct_volume(self, shape=(5, 5, 3)):
        """
        Simulate a more realistic anatomical CT volume with smooth variations across slices.
        """
        x = np.linspace(-1, 1, shape[0])
        y = np.linspace(-1, 1, shape[1])
        xv, yv = np.meshgrid(x, y, indexing='ij')

        volume = np.zeros(shape, dtype=np.float32)

        for z in range(shape[2]):
            # Add a Gaussian bump whose intensity and position varies with z
            cx = 0.2 * (z - shape[2] // 2)
            cy = -0.2 * (z - shape[2] // 2)
            gaussian = np.exp(-((xv - cx)**2 + (yv - cy)**2) * 10)
            intensity_scale = 0.5 + 0.1 * z  # simulate deeper tissues being denser
            volume[:, :, z] = gaussian * intensity_scale * 255  # scale to HU-like range

        return volume

    # Step 1 – Create a simple 3D CT volume
    def create_ct_volume(self, shape=(3, 3, 2)):
        """Create a simple 3D array to simulate a CT volume."""
        return np.arange(np.prod(shape)).reshape(shape)

    # Step 2 – Print volume metadata
    def print_volume_info(self, volume):

        print(volume)

        print("\n CT Volume Metadata:")
        print("- Volume shape:", volume.shape)
        print("- Sagittal slices (x):", volume.shape[0])
        print("- Coronal slices  (y):", volume.shape[1])
        print("- Axial slices    (z):", volume.shape[2])


    # Step 3 – Show the middle slices from each plane
    def show_middle_slices(self, volume):
        x = volume.shape[0] // 2
        y = volume.shape[1] // 2
        z = volume.shape[2] // 2

        print("\n🧠 Middle Slices:")
        print("Sagittal slice shape:", volume[x, :, :].shape)
        print("Coronal slice shape:", volume[:, y, :].shape)
        print("Axial slice shape:", volume[:, :, z].shape)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(volume[x, :, :], cmap='gray')
        axes[0].set_title('Sagittal View\nDivides left & right')

        axes[1].imshow(volume[:, y, :], cmap='gray')
        axes[1].set_title('Coronal View\nDivides front & back')

        axes[2].imshow(volume[:, :, z], cmap='gray')
        axes[2].set_title('Axial (Transverse) View\nDivides top & bottom')

        plt.tight_layout()
        plt.show()


    # Step 4 – View arbitrary slice from any plane
    def view_slice(self, volume, plane='axial', index=0):
        """
        Display an arbitrary slice from a given anatomical plane.
        """
        if plane == 'axial':
            slice_ = volume[:, :, index]
            title = f'Axial (z=fixed, [:, :, {index}], slices={volume.shape[2]}) - Divides top & bottom'
        elif plane == 'coronal':
            slice_ = volume[:, index, :]
            title = f'Coronal (y=fixed, [:, {index}, :], slices={volume.shape[1]}) - Divides front & back'
        elif plane == 'sagittal':
            slice_ = volume[index, :, :]
            title = f'Sagittal (x=fixed, [{index}, :, :], slices={volume.shape[0]}) - Divides left & right'
        else:
            raise ValueError("Plane must be one of: 'axial', 'coronal', 'sagittal'")

        print(f"Viewing {title}")
        print("Slice shape:", slice_.shape)
        print(slice_)
        # plt.imshow(slice_, cmap='gray')
        # plt.title(title)
        # plt.axis('off')
        # plt.show()


# # Step 6 – Main test driver
# def main():
#
#     # volume = create_ct_volume(shape=(3, 3, 2))  # Feel free to change dimensions
#     volume = generate_realistic_ct_volume()
#
#     print_volume_info(volume)
#
#     # show_middle_slices(volume)
#     view_slice(volume, 'axial', 0)
#     view_slice(volume, 'coronal', 0)
#     view_slice(volume, 'sagittal', 0)
#     # interactive_slice_viewer(volume)
#
#     # Display with more prominent voxel highlight
#     plot_voxel_planes_highlighted(volume_shape=(10, 10, 6), voxel_coords=(5, 5, 3))
#
#
# if __name__ == '__main__':
#     main()
