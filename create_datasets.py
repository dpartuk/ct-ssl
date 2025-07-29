import os
#import time
from pathlib import Path

import ct_config
from ct_image_processing import ImageProcessing

def main():
    job_dir = os.path.dirname(os.path.realpath(__file__))
    print(f'job dir: {job_dir}')

    root = 'ct_images/'
    number_of_ct_patients = 131

    liver_images_path = f"{root}Task03_Liver/imagesTr/"
    liver_labels_path = f"{root}Task03_Liver/labelsTr/"

    path_to_dataset = f"{job_dir}/datasets/liver_dataset_{number_of_ct_patients}.npz"

    imgProcessor = ImageProcessing()

    X_all = []
    Y_all = []

    print(f"Start Building CT Dataset with {number_of_ct_patients} patients")

    X_all, Y_all, patient_ids, total = imgProcessor.create_dataset(liver_images_path,
                                                                   liver_labels_path,
                                                                   binary=True,
                                                                   target_size=(256, 256),
                                                                   hu_window=(30, 180),
                                                                   number_of_ct_patients=number_of_ct_patients,
                                                                   labeled_only=True)

    print(total)
    print('Len (X, Y, Patients):', len(X_all), len(Y_all), len(patient_ids))
    print(f'Sample Patient Shapes ({patient_ids[2]}): X[2] Y[2]:', X_all[2].shape, Y_all[2].shape)

    # Save Dataset

    if (len(X_all) > 0 and len(Y_all) > 0):
        imgProcessor.save_dataset(X_all, Y_all, patient_ids, f"datasets/liver_dataset_{number_of_ct_patients}.npz")
    else:
        print('X_all || Y_all are empty')

if __name__ == '__main__':
        main()
