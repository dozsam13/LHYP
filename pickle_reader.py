import sys
import pickle
from domain.patient_data import PatientData
import cv2
import numpy as np

fn = sys.argv[1]
patient_data = None
with (open(fn, "rb")) as openfile:
  while True:
      try:
          patient_data = pickle.load(openfile)
      except EOFError:
          break

cv2.imwrite( "bottom.jpg", patient_data.contour_diff_matricies[0] );
cv2.imwrite( "mid.jpg", patient_data.contour_diff_matricies[1] );
cv2.imwrite( "top.jpg", patient_data.contour_diff_matricies[2] );