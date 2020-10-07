import pickle
import os
import numpy as np

class DataReader:
    def __init__(self, path):
        patient_file_paths = list(map(lambda f: os.path.join(path, f), os.listdir(path)))
        self.x = []
        for patient_file_path in patient_file_paths:
            with (open(patient_file_path, "rb")) as patient_file:
                patient_data = pickle.load(patient_file)
                if len(patient_data.diastole_slices) == 3:
                    multi_channel_picture = np.dstack((patient_data.diastole_slices[0], patient_data.diastole_slices[1],
                                                       patient_data.diastole_slices[2]
                                                       ))
                    self.x.append(multi_channel_picture)
                else:
                    print(patient_file_path)