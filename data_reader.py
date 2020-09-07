import pickle
import os
import cv2 as cv
import numpy as np


class DataReader:
    # possible_pathologies = ['U18_m', 'Amyloidosis', 'U18_f', 'EMF', 'Fabry', 'adult_m_sport', 'Aortastenosis',
    # 'Normal', 'HCM', 'adult_f_sport']
    possible_pathologies = ['Normal', 'HCM']

    def __init__(self, path):
        patient_file_paths = list(map(lambda f: os.path.join(path, f), os.listdir(path)))
        self.x = []
        self.y = []
        for patient_file_path in patient_file_paths:
            with (open(patient_file_path, "rb")) as patient_file:
                patient_data = pickle.load(patient_file)
                if len(patient_data.diastole_slices) == 3 and not (patient_data.pathology is None) and patient_data.pathology != "":
                    multi_channel_picture = np.dstack((patient_data.diastole_slices[0], patient_data.diastole_slices[1],
                                                       patient_data.diastole_slices[2]))
                    self.x.append(multi_channel_picture)
                    if not patient_data.pathology is None or patient_data.pathology != "":
                        continue
                    if patient_data.pathology in DataReader.possible_pathologies:
                        self.y.append(DataReader.possible_pathologies.index(patient_data.pathology))
                    else:
                        self.y.append(len(self.possible_pathologies))
                else:
                    print(patient_file_path)

    @staticmethod
    def __resize(picture):
        return cv.resize(picture, (224, 224), interpolation=cv.INTER_AREA)
