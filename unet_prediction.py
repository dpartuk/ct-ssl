from ct_config import debug, epochs, batch_size, number_of_ct_patients

from ct_dataset import CTDataset
from UNet_Model.segmentation_evaluator import SegmentationEvaluator
import numpy as np

class UNETEvaluator:
    def __init__(self, dataset_train, model_runner, finetune = False):
        self.dataset = dataset_train
        self.model_runner = model_runner

        if finetune:
            self.X_test = dataset_train.X_test
            self.Y_test = dataset_train.Y_test

            self.shortcut_sliced_patient_ids(dataset_train.images,
                                             dataset_train.labels,
                                             dataset_train.test_idx,
                                             dataset_train.patient_ids)
        else:
            self.X_test = dataset_train.X_test_masked
            self.Y_test = dataset_train.X_test

            self.shortcut_sliced_patient_ids(dataset_train.images,
                                             dataset_train.labels,
                                             dataset_train.test_idx,
                                             dataset_train.patient_ids)

        # print("total patients for testing:", len(dataset_train.test_idx))
        # patient_ids_test = [dataset_train.patient_ids[i] for i in dataset_train.test_idx]
        # print("patient_ids_test: ", patient_ids_test)
        #
        # max_patients = 30
        # selected_ids = patient_ids_test[:max_patients]
        # print("selected_ids: ", selected_ids)
        # print("X_test.shape: ", self.X_test.shape)
        # selected_X = []
        # selected_X.append(self.X_test)
        # self.slice_patient_ids = [
        #     pid for pid, vol in zip(selected_ids, selected_X) for _ in range(vol.shape[0])
        # ]
        #
        # print("Len slice_patient_ids: ", len(self.slice_patient_ids))
        # print("slice_patient_ids: ", self.slice_patient_ids)



    def evaluate(self):
        print("Evaluating...")
        evaluator = SegmentationEvaluator(self.model_runner.pipeline)

        # Run full eval
        mean_dice, mean_iou, dice_patients, iou_patients = evaluator.evaluate(
            self.X_test, self.Y_test, patient_ids=self.slice_patient_ids
        )

        Y_pred = self.model_runner.pipeline.predict(self.X_test)
        Y_pred_binary = (Y_pred > 0.5).astype(np.float32)

        dice_scores = [evaluator.compute_dice(self.Y_test[i], Y_pred_binary[i])
                       for i in range(len(self.Y_test))]

        evaluator.visualize(self.X_test, self.Y_test, Y_pred_binary, dice_scores, num_examples=8)

    def limit_test_patients(self, X_test_all, Y_test_all, patient_ids, max_patients=1):
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
        if debug:
            print("\n####### Enter")
            print(type(X_test_all))
        selected_X = X_test_all[:max_patients]
        selected_Y = Y_test_all[:max_patients]
        selected_ids = patient_ids[:max_patients]

        X_test = np.concatenate(selected_X, axis=0)
        Y_test = np.concatenate(selected_Y, axis=0)

        if debug:
            print("selected_ids: ", selected_ids)
            print("selected_X.shape: ", X_test.shape)

        slice_patient_ids = [
            pid for pid, vol in zip(selected_ids, selected_X) for _ in range(vol.shape[0])
        ]

        if debug:
            print("Len slice_patient_ids: ", len(slice_patient_ids))
            print("####### Exit\n")

        return X_test, Y_test, slice_patient_ids

    def shortcut_sliced_patient_ids(self, X_all, Y_all, test_idx, patient_ids):
        print("total patients for testing:", len(test_idx))

        X_test_all = [X_all[i] for i in test_idx]
        Y_test_all = [Y_all[i] for i in test_idx]
        patient_ids_test = [patient_ids[i] for i in test_idx]


        if debug:
            print("X Test All Shape: ", len(X_test_all), "Y Test All Shape: ", len(Y_test_all), "patient_ids_test: ",
                  len(patient_ids_test))
            print("test_idx: ", test_idx)
            print("patient_ids_test: ", patient_ids_test)

        X_test_limited, Y_test_limited, self.slice_patient_ids = self.limit_test_patients(X_test_all, Y_test_all,
                                                                                patient_ids_test, max_patients=len(test_idx))

        # print(X_test_limited.shape, Y_test_limited.shape, len(self.slice_patient_ids))
        # print("slice_patient_ids: ", self.slice_patient_ids)