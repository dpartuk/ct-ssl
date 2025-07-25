import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from ct_config import debug

# For Prediction Purpose
class SegmentationEvaluator:
    def __init__(self, pipeline, threshold=0.5, epsilon=1e-6):
        """
        Initialize the evaluator with a trained segmentation pipeline.

        Args:
            pipeline: Trained UNetSegmentationPipeline instance.
            threshold: Threshold for binarizing predicted masks.
            epsilon: Smoothing term for Dice/IoU calculations.
        """
        self.pipeline = pipeline
        self.threshold = threshold
        self.epsilon = epsilon

    def compute_dice(self, y_true, y_pred):
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred)
        return (2. * intersection + self.epsilon) / (union + self.epsilon)

    def compute_iou(self, y_true, y_pred):
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection
        return (intersection + self.epsilon) / (union + self.epsilon)

    def evaluate(self, X_test, Y_test, patient_ids=None):
        """
        Predict and evaluate segmentation masks.

        Args:
            X_test (np.ndarray): Test images of shape (N, H, W, 1)
            Y_test (np.ndarray): Ground truth masks of shape (N, H, W, 1)
            patient_ids (list or np.ndarray): Optional. Patient ID for each slice.

        Returns:
            mean_dice_per_slice, mean_iou_per_slice, mean_dice_per_patient, mean_iou_per_patient
        """
        Y_test = Y_test.astype(np.float32)
        Y_pred = self.pipeline.predict(X_test)
        Y_pred_binary = (Y_pred > 0.5).astype(np.float32)

        dice_scores = [self.compute_dice(Y_test[i], Y_pred_binary[i]) for i in range(len(Y_test))]
        iou_scores = [self.compute_iou(Y_test[i], Y_pred_binary[i]) for i in range(len(Y_test))]

        mean_dice = np.mean(dice_scores)
        mean_iou = np.mean(iou_scores)

        print(f"\nMean Dice score per slice: {mean_dice:.4f}")
        print(f"Mean IoU  score per slice: {mean_iou:.4f}")

        mean_dice_per_patient, mean_iou_per_patient = None, None

        if patient_ids is not None:
            dice_by_patient = defaultdict(list)
            iou_by_patient = defaultdict(list)
            for i in range(len(Y_test)):
                pid = patient_ids[i]
                dice_by_patient[pid].append(dice_scores[i])
                iou_by_patient[pid].append(iou_scores[i])
            mean_dice_per_patient = [np.mean(v) for v in dice_by_patient.values()]
            mean_iou_per_patient = [np.mean(v) for v in iou_by_patient.values()]
            print(f"Mean Dice score per patient: {np.mean(mean_dice_per_patient):.4f}")
            print(f"Mean IoU  score per patient: {np.mean(mean_iou_per_patient):.4f}")

        return mean_dice, mean_iou, mean_dice_per_patient, mean_iou_per_patient

    def visualize(self, X, Y_true, Y_pred_binary, dice_scores, num_examples=5, seed=None):
        """
        Visualize predictions alongside ground truth and overlays.

        Args:
            X (np.ndarray): Input CT images.
            Y_true (np.ndarray): Ground truth segmentation masks.
            Y_pred_binary (np.ndarray): Binary predicted masks.
            dice_scores (list): Dice scores per slice.
            num_examples (int): Number of examples to show.
            seed (int or None): Random seed for reproducibility.
        """
        ppid = os.getppid()
        if seed is not None:
            np.random.seed(seed)

        indices = np.random.choice(len(X), num_examples, replace=False)
        plt.figure(figsize=(12, num_examples * 3))
        for i, idx in enumerate(indices):
            plt.subplot(num_examples, 4, i * 4 + 1)
            plt.imshow(X[idx].squeeze(), cmap='gray')
            plt.title("CT Slice")
            plt.axis('off')

            plt.subplot(num_examples, 4, i * 4 + 2)
            plt.imshow(Y_true[idx].squeeze(), cmap='gray')
            plt.title("Ground Truth")
            plt.axis('off')

            plt.subplot(num_examples, 4, i * 4 + 3)
            plt.imshow(Y_pred_binary[idx].squeeze(), cmap='gray')
            plt.title(f"Prediction\nDice: {dice_scores[idx]:.3f}")
            plt.axis('off')

            plt.subplot(num_examples, 4, i * 4 + 4)
            plt.imshow(X[idx].squeeze(), cmap='gray')
            plt.imshow(Y_pred_binary[idx].squeeze(), alpha=0.3, cmap='Reds')
            plt.title("Overlay")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'dice_scores_output_{ppid}.png')
        if debug:
            plt.show()
