import os
import pickle
import sys
import json

path = sys.argv[1]
patient_file_paths = list(map(lambda f : os.path.join(path, f), os.listdir(path)))
pathology_dict = {}
pathologies = {'Normal', 'HCM'}
for patient_file_path in patient_file_paths:
  with (open(patient_file_path, "rb")) as patient_file:
    while True:
      try:
        patient_data = pickle.load(patient_file)
        if not patient_data.pathology in pathologies.keys():
            patient_data.pathology = 'Other'
        pathology_dict[patient_data.scan_id] = patient_data.pathology
      except EOFError:
        break

path2 = sys.argv[2]
with open(path2) as f:
  occurrence_dict = json.load(f)

x = {'Normal' : 0, 'HCM' : 0, 'Other' : 0}
for key in occurrence_dict:
    pathology = pathology_dict[key]
    n = occurrence_dict[key]
    x[pathology] += n

print(x)
