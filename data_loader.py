import sys
import pickle
import os


class DataLoader:
  def __init__(self, path):
    patient_file_paths = list(map(lambda f : os.path.join(path, f), os.listdir(path)))
    self.x = []
    self.y = []
    for patient_file_path in patient_file_paths:
      with (open(patient_file_path, "rb")) as patient_file:
        while True:
          try:
            patient_data = pickle.load(patient_file)
            self.x.append(patient_data.diastole[1][3])
            self.y.append(int(patient_data.pathology == 'HCM'))
          except EOFError:
            break