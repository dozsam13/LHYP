import pickle
import sys

import cv2

from ssl.puzzle_shuffle import PuzzleShuffle

fn = sys.argv[1]
patient_data = None
img_shuffle = PuzzleShuffle(3)
with (open(fn, "rb")) as openfile:
  while True:
      try:
          patient_data = pickle.load(openfile)
      except EOFError:
          break
cv2.imwrite("d_bottom.jpg", patient_data.diastole_slices[0])
cv2.imwrite("d_mid.jpg", patient_data.diastole_slices[1])
cv2.imwrite("d_top.jpg", patient_data.diastole_slices[2])