import argparse

# import torchvision.transforms as transforms
# import torchvision.datasets as datasets

import os
#import time
from pathlib import Path

from ct_dataset import CTDataset
import ct_config
# from unet_runner import UNETRunner
# from unet_prediction import UNETEvaluator
from UNet_Model.unet_segmentation_pipeline import UNetSegmentationPipeline
from UNet_Model.orig_segmentation_evaluator import SegmentationEvaluator

import matplotlib.pyplot as plt
import numpy as np
# from ct_config import debug
# from ct_masking import CTMask

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')

    parser.add_argument('--patch_size', default=16, type=int,
                        help='masking patch size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    # Dataset parameters
    parser.add_argument('--data_path', default='datasets/', type=str,
                        help='relative path to dataset')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')

    # parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    parser.add_argument('--number_of_ct_patients', default=5, type=int,
                        choices=[5, 10, 25, 50, 131], help='5, 10, 25, 50, 131')

    parser.add_argument('--debug', action='store_true')

    return parser

def plot_training_history(history_dict, metrics=("loss", "dice_coef", "iou_metric")):
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
    if ct_config.debug:
        plt.show()

def limit_test_patients(X_test_all, Y_test_all, patient_ids, max_patients=1):
    """
    Limit test data to the first `max_patients` and return concatenated slices with patient IDs.

    Args:
        X_test_all (list of np.ndarray): Test CT volumes per patient
        Y_test_all (list of np.ndarray): Test segmentation masks per patient
        patient_ids (list of str): Patient IDs corresponding to each volume
        max_patients (int): Max number of patients to include in test set

    Returns:
        X_test (np.ndarray): Flattened test images from selected patients
        Y_test (np.ndarray): Flattened test masks from selected patients
        slice_patient_ids (list of str): Slice-level patient ID list
    """
    selected_X = X_test_all[:max_patients]
    selected_Y = Y_test_all[:max_patients]
    selected_ids = patient_ids[:max_patients]

    X_test = np.concatenate(selected_X, axis=0)
    Y_test = np.concatenate(selected_Y, axis=0)

    slice_patient_ids = [
        pid for pid, vol in zip(selected_ids, selected_X) for _ in range(vol.shape[0])
    ]

    return X_test, Y_test, slice_patient_ids

def main(args):

    if args.debug:
        ct_config.debug = True

    ppid = os.getppid()
    job_dir = os.path.dirname(os.path.realpath(__file__))
    print(f'job dir: {job_dir}')

    path_to_dataset = f"{job_dir}/{args.data_path}/liver_dataset_{ct_config.number_of_ct_patients}.npz"
    dataset = CTDataset(path_to_dataset)
    dataset.split_supervised(ct_config.number_of_ct_patients)
    if ct_config.print_smaples:
        dataset.print_samples()

    # Train
    from UNet_Model.unet_segmentation_pipeline import UNetSegmentationPipeline
    pipeline = UNetSegmentationPipeline(input_shape=(256, 256, 1))
    pipeline.summary()

    history = pipeline.fit(dataset.X_train, dataset.Y_train, dataset.X_val, dataset.Y_val,
                           epochs=ct_config.epochs, batch_size=ct_config.batch_size, verbose=2)

    test_score = pipeline.evaluate(dataset.X_val, dataset.Y_val)
    print("Validation Dice and IoU:", test_score)

    # Saving the pipeline model
    model_name = f"UNet_Model/saved_models/unet_supervised_ct_liver_{ct_config.number_of_ct_patients}_{ct_config.epochs}.keras"
    file_history_name = f"UNet_Model/saved_models/unet_supervised_ct_history_{ct_config.number_of_ct_patients}_{ct_config.epochs}"
    pipeline.save(model_name)
    pipeline.save_training_history(history, file_history_name, format="json")

    # Reload history
    history_dict = pipeline.load_training_history(file_history_name, format="json")
    plot_training_history(history_dict, metrics=("loss", "dice_coef", "iou_metric"))

    pipeline = UNetSegmentationPipeline.load(model_name)
    pipeline.summary()

    # Prediction
    len(dataset.X_test), len(dataset.Y_test), len(dataset.patient_ids)

    print("total patients for testing:", len(dataset.test_idx))

    X_test_all = [dataset.images[i] for i in dataset.test_idx]
    Y_test_all = [dataset.labels[i] for i in dataset.test_idx]
    patient_ids_test = [dataset.patient_ids[i] for i in dataset.test_idx]
    print("X Test All Shape: ", len(X_test_all), "Y Test All Shape: ", len(Y_test_all), "patient_ids_test: ",
          len(patient_ids_test))
    print("test_idx: ", dataset.test_idx)
    print("patient_ids_test: ", patient_ids_test)

    X_test_limited, Y_test_limited, slice_patient_ids = limit_test_patients(X_test_all, Y_test_all, patient_ids_test,
                                                                            max_patients=5)

    print(X_test_limited.shape, Y_test_limited.shape, len(slice_patient_ids))
    print("slice_patient_ids: ", slice_patient_ids)

    evaluator = SegmentationEvaluator(pipeline)

    # Run full eval
    mean_dice, mean_iou, dice_patients, iou_patients = evaluator.evaluate(
        X_test_limited, Y_test_limited, patient_ids=slice_patient_ids
    )

    # evaluator = SegmentationEvaluator(pipeline)
    #
    # # Run full eval
    # mean_dice, mean_iou, dice_patients, iou_patients = evaluator.evaluate(
    #     X_test_limited, Y_test_limited, patient_ids=slice_patient_ids
    # )

    Y_pred = pipeline.predict(X_test_limited)
    Y_pred_binary = (Y_pred > 0.5).astype(np.float32)

    dice_scores = [evaluator.compute_dice(Y_test_limited[i], Y_pred_binary[i])
                   for i in range(len(Y_test_limited))]

    evaluator.visualize(X_test_limited, Y_test_limited, Y_pred_binary, dice_scores, num_examples=8)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
