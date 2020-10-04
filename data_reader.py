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
                if len(patient_data.hearth_cycle) != 0 and not self.empty(patient_data.pathology):
                    patient_data.hearth_cycle += [np.zeros((110,110)) for i in range(25-len(patient_data.hearth_cycle))]
                    self.x.append(patient_data.hearth_cycle)
                    if patient_data.pathology in DataReader.possible_pathologies:
                        self.y.append(DataReader.possible_pathologies.index(patient_data.pathology))
                    else:
                        self.y.append(len(self.possible_pathologies))
                else:
                    print(patient_file_path)

    def empty(self, pathology):
        return pathology is None or pathology == ""