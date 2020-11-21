import pickle
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

def createJpg(pickle_path, to_path):
    patient_data = None
    with (open(pickle_path, "rb")) as openfile:
        while True:
            try:
                patient_data = pickle.load(openfile)
            except EOFError:
                break
    img_path_prefix = os.path.join(to_path, patient_data.scan_id)
    plt.imsave(img_path_prefix + "_bottom.jpg", patient_data.diastole_slices[0], cmap = "gray")
    plt.imsave(img_path_prefix + "_mid.jpg", patient_data.diastole_slices[1], cmap = "gray")
    plt.imsave(img_path_prefix + "_top.jpg", patient_data.diastole_slices[2], cmap = "gray")


in_dir = sys.argv[1]
out_dir = sys.argv[2]
pickle_files = os.listdir(in_dir)
np.random.shuffle(pickle_files)
for picke_file in pickle_files[:100]:
    createJpg(os.path.join(in_dir, picke_file), out_dir)