import argparse

# import torchvision.transforms as transforms
# import torchvision.datasets as datasets

import os
#import time
from pathlib import Path

from ct_dataset import CTDataset
import ct_config
from unet_runner import UNETRunner
from unet_prediction import UNETEvaluator


# from ct_config import debug
# from ct_masking import CTMask

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning', add_help=False)

    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')

    # Dataset parameters
    parser.add_argument('--data_path', default='datasets/', type=str,
                        help='relative path to dataset')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')

    parser.add_argument('--number_of_ct_patients', default=5, type=int,
                        choices=[5, 10, 25, 50, 131], help='5, 10, 25, 50, 131')

    parser.add_argument('--debug', action='store_true')

    return parser

def main(args):

    if args.debug:
        ct_config.debug = True

    job_dir = os.path.dirname(os.path.realpath(__file__))
    print(f'job dir: {job_dir}')

    # Load dataset and split to train, validation, test
    path_to_dataset = f"{job_dir}/{args.data_path}/liver_dataset_{ct_config.number_of_ct_patients}.npz"
    dataset = CTDataset(path_to_dataset)
    dataset.split_finetune(ct_config.number_of_ct_patients)
    if ct_config.print_smaples:
        dataset.print_samples()

    # Training
    model_runner = UNETRunner(dataset)
    model_runner.load_model()
    model_runner.train(finetune = True)
    model_runner.save_run(finetune = True)
    model_runner.load_model(finetune = True)

    # Prediction
    model_evaluator = UNETEvaluator(dataset, model_runner, finetune = True)
    model_evaluator.evaluate()




if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
