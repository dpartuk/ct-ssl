import os
#import time
import numpy as np
from pathlib import Path
from ct_image_processing_pred import ImageProcessing
from UNet_Model.unet_segmentation_pipeline import UNetSegmentationPipeline
from UNet_Model.segmentation_evaluator import SegmentationEvaluator
from ct_visualizer import CTVisualizer

def main():
    number_of_ct_patients = 131
    epoch_num = 20
    model_name = f"UNet_Model/saved_models/unet_finetune_ct_liver_{number_of_ct_patients}_{epoch_num}.keras"

    pipeline = UNetSegmentationPipeline.load(model_name)
    pipeline.summary()



    root = 'ct_images/' #'/Users/yigal/CT-Datasets/'

    liver_images_path = f"{root}Task03_Liver/imagesTr/"
    liver_labels_path = f"{root}Task03_Liver/labelsTr/"

    liver_images_test_path = f"{root}Task03_Liver/imagesTs/"
    liver_labels_test_path = f"{root}Task03_Liver/labelsTs/"

    imgProcessor = ImageProcessing()

    ctVisualizer = CTVisualizer()

    X_all = []
    Y_all = []

    num_test_patients = 10

    X_global_test_list, Y_global_test_list, patient_ids_test, total_test = imgProcessor.create_dataset(
        liver_images_test_path,
        liver_labels_test_path,
        binary=True,
        target_size=(256, 256),
        hu_window=(30, 180),
        number_of_ct_patients=num_test_patients,
        labeled_only=True,
        patient_offset=121)

    print(total_test)

    total_slices = sum(x.shape[0] for x in X_global_test_list)
    print(f"Total 2D slices: {total_slices}")

    X_global_test = np.concatenate(X_global_test_list, axis=0)
    Y_global_test = np.concatenate(Y_global_test_list, axis=0)

    print(X_global_test.shape, Y_global_test.shape)

    if (len(X_global_test_list) > 0 and len(Y_global_test_list) > 0):
        imgProcessor.save_dataset(X_global_test_list, Y_global_test_list, patient_ids_test,
                                  f"datasets/test_liver_dataset_{num_test_patients}.npz")
    else:
        print('X_all || Y_all are empty')

    Y_pred = pipeline.predict(X_global_test)

    Y_pred_binary = (Y_pred > 0.5).astype(np.float32)

    evaluator = SegmentationEvaluator(pipeline)

    dice_scores = [evaluator.compute_dice(Y_global_test[i], Y_pred_binary[i])
                   for i in range(len(Y_global_test))]

    avg_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)

    print(f"AVG dice score: {avg_dice:.4f} ± {std_dice:.4f}")

    iou_scores = [evaluator.compute_iou(Y_global_test[i], Y_pred_binary[i])
                  for i in range(len(Y_global_test))]

    avg_iou = np.mean(iou_scores)
    std_iou = np.std(iou_scores)

    print(f"AVG iou score: {avg_iou:.4f} ± {std_iou:.4f}")

    evaluator.visualize(X_global_test, Y_global_test, Y_pred_binary, dice_scores, num_examples=10, seed=42)

    global_scores = [evaluator.compute_global_metrics(Y_global_test[i], Y_pred_binary[i])
                     for i in range(len(Y_global_test))]

    # Initialize dictionary for accumulating sums
    metric_sums = {key: 0.0 for key in global_scores[0]}

    # Sum all values
    for score in global_scores:
        for key in score:
            metric_sums[key] += score[key]

    # Average
    num_items = len(global_scores)
    avg_scores = {key: metric_sums[key] / num_items for key in metric_sums}

    # Print result
    print("Average metrics over all slices:")
    for metric, value in avg_scores.items():
        print(f"{metric}: {value:.4f}")



if __name__ == '__main__':
    main()