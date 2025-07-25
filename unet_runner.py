import os
import matplotlib.pyplot as plt
from ct_dataset import CTDataset
from UNet_Model.unet_segmentation_pipeline import UNetSegmentationPipeline
# from UNet_Model.unet_segmentation_pipeline import UNetSegmentationPipeline

from ct_config import debug, epochs, finetune_epochs ,batch_size, number_of_ct_patients

class UNETRunner:
    def __init__(self, dataset_train):
        # Build and train model
        self.pipeline = UNetSegmentationPipeline(input_shape=(256, 256, 1))

        self.pipeline.summary()

        self.history = None
        self.dataset = dataset_train

    def train(self, finetune=False):
        # epochs = 100
        # batch_size = 128
        if finetune:
            X_train = self.dataset.X_train
            Y_train = self.dataset.Y_train
            # if True: # REMOVE this section after finishing development of prediction. Using smaller dataset
            #     X_train = self.dataset.X_test
            #     Y_train = self.dataset.Y_test
            X_val = self.dataset.X_val
            Y_val = self.dataset.Y_val
            epoch_num = finetune_epochs
        else:
            X_train = self.dataset.X_train_masked
            Y_train = self.dataset.X_train
            X_val = self.dataset.X_val_masked
            Y_val = self.dataset.X_val
            epoch_num = epochs

        self.history = self.pipeline.fit(X_train, Y_train, X_val, Y_val, epochs=epoch_num, batch_size=batch_size, verbose=2)

        test_score = self.pipeline.evaluate(X_val, Y_val)
        print("Validation Dice and IoU:", test_score)

    def save_run(self, finetune=False):
        # Saving the pipeline model
        if finetune:
            epoch_num = finetune_epochs
            model_name = f"UNet_Model/saved_models/unet_finetune_ct_liver_{number_of_ct_patients}_{epoch_num}.keras"
            file_history_name = f"UNet_Model/saved_models/unet_finetune_ct_history_{number_of_ct_patients}_{epoch_num}"
        else:
            epoch_num = epochs
            model_name = f"UNet_Model/saved_models/unet_ssl_ct_liver_{number_of_ct_patients}_{epoch_num}.keras"
            file_history_name = f"UNet_Model/saved_models/unet_ssl_ct_history_{number_of_ct_patients}_{epoch_num}"

        self.pipeline.save(model_name)
        self.pipeline.save_training_history(self.history, file_history_name, format="json")

        history_dict = self.pipeline.load_training_history(file_history_name, format="json")
        self.plot_training_history(history_dict, metrics=("loss", "dice_coef", "iou_metric"))


    def load_model(self, finetune=False):
        # Probably already in memory, but maintaining consistency with notebook
        if finetune:
            epoch_num = finetune_epochs
            model_name = f"UNet_Model/saved_models/unet_finetune_ct_liver_{number_of_ct_patients}_{epoch_num}.keras"
        else:
            epoch_num = epochs
            model_name = f"UNet_Model/saved_models/unet_ssl_ct_liver_{number_of_ct_patients}_{epoch_num}.keras"

        self.pipeline = UNetSegmentationPipeline.load(model_name)
        self.pipeline.summary()

    def plot_training_history(self, history_dict, metrics=("loss", "dice_coef", "iou_metric")):
        """
        Plot training and validation curves for selected metrics.

        Args:
            history_dict: Dict returned from model.history or load_training_history()
            metrics: Tuple of metric names to plot
        """
        for metric in metrics:
            if metric in history_dict:
                plt.plot(history_dict[metric], label=f"Train {metric}")
            val_key = f"val_{metric}"
            if val_key in history_dict:
                plt.plot(history_dict[val_key], label=f"Val {metric}")

        ppid = os.getppid()
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training History")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'training_history_output_{ppid}.png')
        if debug:
            plt.show()

