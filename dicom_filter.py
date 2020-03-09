from con_reader import CONreaderVM
from dicom_reader import DCMreaderVM
from utils import get_logger
from math import floor
import numpy as np
import pickle
import os
import sys

logger = get_logger(__name__)

def create_path_for_file(pickle_file_path):
    os.makedirs(os.path.dirname(pickle_file_path), exist_ok=True)

def collect_contour_slices_for_frames(contours):
    frameSliceDict = {}
    for slc in contours:
        for frm in contours[slc]:
            if not(frm in frameSliceDict):
                frameSliceDict[frm] = []
            frameSliceDict[frm].append(slc)
    return frameSliceDict

def create_pickle_for_patient(in_dir, out_dir):
    scan_id = os.path.basename(in_dir)
    image_folder = os.path.join(in_dir, "sa", "images")
    con_file = os.path.join(in_dir, "sa", "contours.con")

    if not os.path.isdir(image_folder):
        logger.error("Could not find image folder for: {}".format(scan_id))
        return
    if not os.path.isfile(con_file):
        logger.error("Could not find .con fil for: {}".format(scan_id))
        return

    dr = DCMreaderVM(image_folder)
    if dr.num_frames == 0 and dr.num_frames == 0 or dr.broken:
        logger.error("Could not create pickle file for {}".format(scan_id))
        return

    cr = CONreaderVM(con_file)
    contours = cr.get_hierarchical_contours()

    frameSliceDict = collect_contour_slices_for_frames(contours)
    pickle_file_path = os.path.join(out_dir, scan_id + ".p")
    create_path_for_file(pickle_file_path)
    with (open(pickle_file_path, "wb")) as pickleFile:
        pickle.dump((scan_id, cr.get_volume_data()), pickleFile)
        for frm in frameSliceDict:
            slices = frameSliceDict[frm]
            filteredSliceIndexes =  np.percentile(np.array(slices), (19,50,83))
            filteredSliceIndexes = [floor(i) for i in filteredSliceIndexes]

            for slc in filteredSliceIndexes:
                image = dr.get_image(slc,frm)
                pickle.dump((image.astype('uint8'), contours[slc][frm]), pickleFile)

in_dir = sys.argv[1]
out_dir = sys.argv[2]

if not os.path.isdir(in_dir):
    logger.error("Invalid input directory: {}".format(in_dir))
else:
    patient_folders = sorted(os.listdir(in_dir))
    for patient_folder in patient_folders:
        create_pickle_for_patient(os.path.join(in_dir, patient_folder), out_dir)
