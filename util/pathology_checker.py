import os
import pickle
import sys

path = sys.argv[1]
patient_file_paths = list(map(lambda f : os.path.join(path, f), os.listdir(path)))
x = {'Normal' : 0, 'HCM' : 0, 'Other' : 0}
for patient_file_path in patient_file_paths:
  with (open(patient_file_path, "rb")) as patient_file:
    while True:
      try:
        patient_data = pickle.load(patient_file)
        if len(patient_data.contour_diff_matricies) == 3:
          if patient_data.pathology in x.keys():
            x[patient_data.pathology] += 1
          else:
            x['Other'] += 1
        else:
          print(patient_file_path)
      except EOFError:
        break

print(x)