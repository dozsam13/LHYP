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

cv2.imwrite( "s_bottom.jpg", patient_data.systole_slices[0] );
cv2.imwrite( "s_mid.jpg", patient_data.systole_slices[1] );
cv2.imwrite( "s_top.jpg", patient_data.systole_slices[2] );
cv2.imwrite( "d_bottom.jpg", patient_data.diastole_slices[0] );
cv2.imwrite( "d_mid.jpg", patient_data.diastole_slices[1] );
cv2.imwrite( "d_top.jpg", patient_data.diastole_slices[2] );