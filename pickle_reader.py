import sys
import pickle
import cv2

fn = sys.argv[1]
patient_data = None
with (open(fn, "rb")) as openfile:
  while True:
      try:
          patient_data = pickle.load(openfile)
      except EOFError:
          break

for i in range(25):
    cv2.imwrite(str(i) + ".jpg", patient_data.hearth_cycle[i])