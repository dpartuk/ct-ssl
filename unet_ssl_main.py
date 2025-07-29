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

def set_seed_for_gpu(seed=42):
    import os, random, numpy as np, tensorflow as tf
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

def main(args):

    if args.debug:
        ct_config.debug = True

    set_seed_for_gpu()

    job_dir = os.path.dirname(os.path.realpath(__file__))
    print(f'job dir: {job_dir}')
    # print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # print("{}".format(args).replace(', ', ',\n'))

    # Load dataset and split to train, validation, test
    path_to_dataset = f"{job_dir}/{args.data_path}/liver_dataset_{ct_config.number_of_ct_patients}.npz"
    dataset_train = CTDataset(path_to_dataset)
    dataset_train.split_train_test(ct_config.number_of_ct_patients)
    dataset_train.mask_and_save()
    dataset_train.print_summary()
    if ct_config.print_smaples:
        dataset_train.print_XY_samples()

    # Training
    model_runner = UNETRunner(dataset_train)
    model_runner.train()
    model_runner.save_run()
    # model_runner.load_model()

    # Prediction
    model_evaluator = UNETEvaluator(dataset_train, model_runner)
    model_evaluator.evaluate()




if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
