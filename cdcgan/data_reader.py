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
                if len(patient_data.diastole_slices) == 3 and not self.empty(patient_data.pathology):
                    multi_channel_picture = np.dstack((patient_data.diastole_slices[0], patient_data.diastole_slices[1],
                                                       patient_data.diastole_slices[2]
                                                       ))
                    self.x.append(multi_channel_picture)
                    y_data = [0 for i in range(len(self.possible_pathologies) + 1)]
                    y_index = 0
                    if patient_data.pathology in DataReader.possible_pathologies:
                        y_index = DataReader.possible_pathologies.index(patient_data.pathology)
                    else:
                        y_index = len(self.possible_pathologies)
                    y_data[y_index] = 1
                    self.y.append(y_data)
                else:
                    print(patient_file_path)

    def empty(self, pathology):
        return pathology is None or pathology == ""
