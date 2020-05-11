import sys
import pickle
import os
import numpy as np
import cv2 as cv


class DataReader:
  possible_pathologies = ['U18_m', 'Amyloidosis', 'U18_f', 'EMF', 'Fabry', 'adult_m_sport', 'Aortastenosis', 'Normal', 'HCM', 'adult_f_sport']
  def __init__(self, path):
    patient_file_paths = list(map(lambda f : os.path.join(path, f), os.listdir(path)))
    self.x = []
    self.y = []
    for patient_file_path in patient_file_paths:
      with (open(patient_file_path, "rb")) as patient_file:
        while True:
          try:
            patient_data = pickle.load(patient_file)
            patient_data.contour_diff_matricies = list(map(self.__resize, patient_data.contour_diff_matricies))
            if len(patient_data.contour_diff_matricies) == 3:
              multi_channel_picture = np.expand_dims(patient_data.contour_diff_matricies[0], axis=0)
              multi_channel_picture = np.append(multi_channel_picture, np.expand_dims(patient_data.contour_diff_matricies[1], axis=0), axis=0)
              multi_channel_picture = np.append(multi_channel_picture, np.expand_dims(patient_data.contour_diff_matricies[2], axis=0), axis=0)
              pathology_vector = [0]*len(DataReader.possible_pathologies)
              pathology_vector[DataReader.possible_pathologies.index(patient_data.pathology)] = 1
              self.x.append(multi_channel_picture)
              self.y.append(pathology_vector)
            else:
              print(patient_file_path)
          except EOFError:
            break

  @staticmethod
  def __resize(picture):
    return cv.resize(picture, (224,224), interpolation = cv.INTER_AREA)
    