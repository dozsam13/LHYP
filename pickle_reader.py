import sys
import pickle
import cv2
from ssl.puzzle_shuffle import PuzzleShuffle
import numpy as np

fn = sys.argv[1]
patient_data = None
img_shuffle = PuzzleShuffle(2, 110)
with (open(fn, "rb")) as openfile:
  while True:
      try:
          patient_data = pickle.load(openfile)
      except EOFError:
          break
cv2.imwrite("d_bottom_norm.jpg", patient_data.diastole_slices[0])
cv2.imwrite("d_mid_norm.jpg", patient_data.diastole_slices[1])
cv2.imwrite("d_top_norm.jpg", patient_data.diastole_slices[2])
img, ind = img_shuffle(np.dstack((patient_data.diastole_slices[0], patient_data.diastole_slices[1], patient_data.diastole_slices[2])))
print(ind)

cv2.imwrite("d_bottom.jpg", img[:, :, 0])
cv2.imwrite("d_mid.jpg", img[:, :, 1])
cv2.imwrite("d_top.jpg", img[:, :, 2])