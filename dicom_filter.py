from con_reader import CONreaderVM
from dicom_reader import DCMreaderVM
from utils import get_logger
from math import floor
import numpy as np
import pickle
import os
import sys
import cv2 as cv

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

def keepLeftVentricleContours(contours):
    left_ventricle_color_modes = {"ln", "lp"}
    left_ventricle_contours = {}
    for slc, frames in contours.items():
        for frm, modes in frames.items():
            filtered_contours = { color_mode: contours[slc][frm][color_mode] for color_mode in left_ventricle_color_modes.intersection(set(modes.keys())) }
            if len(filtered_contours) == 0:
                continue
            if not(slc in left_ventricle_contours):
                left_ventricle_contours[slc] = {}
            left_ventricle_contours[slc][frm] = filtered_contours
    return left_ventricle_contours



def create_pickle_for_patient(in_dir, out_dir):
    scan_id = os.path.basename(in_dir)
    image_folder = os.path.join(in_dir, "sa", "images")
    con_file = os.path.join(in_dir, "sa", "contours.con")

    if not os.path.isdir(image_folder):
        logger.error("Could not find image folder for: {}".format(scan_id))
        return
    if not os.path.isfile(con_file):
        logger.error("Could not find .con file for: {}".format(scan_id))
        return

    dr = DCMreaderVM(image_folder)
    if dr.num_frames == 0 and dr.num_frames == 0 or dr.broken:
        logger.error("Could not create pickle file for {}".format(scan_id))
        return

    cr = CONreaderVM(con_file)
    contours = keepLeftVentricleContours(cr.get_hierarchical_contours())

    frameSliceDict = collect_contour_slices_for_frames(contours)
    pickle_file_path = os.path.join(out_dir, scan_id + ".p")
    create_path_for_file(pickle_file_path)

    startingSlice = max([min(sliceList) for frame, sliceList in frameSliceDict.items()])
    endingSlice = min([max(sliceList) for frame, sliceList in frameSliceDict.items()])
    result = []
    for frm in frameSliceDict:
        s = []

        slices = list(filter(lambda s : s>=startingSlice and s<=endingSlice, frameSliceDict[frm]))
        filteredSliceIndexes =  np.percentile(np.array(slices), (19,50,83))
        filteredSliceIndexes = [floor(i) for i in filteredSliceIndexes]

        for slc in filteredSliceIndexes:
            image = dr.get_image(slc,frm)
            s.append(((image.astype('uint8'), contours[slc][frm], (slc,frm))))
           
        result.append(s)

    comparing_contour_mode = next(iter(set(result[0][1][1].keys()).intersection(result[1][1][1].keys())))

    area1 = cv.contourArea(result[0][1][1][comparing_contour_mode].astype(int))
    area2 = cv.contourArea(result[1][1][1][comparing_contour_mode].astype(int))

    if area2 > area1:
        result[0], result[1] = result[1], result[0]
        print(area2, area1, comparing_contour_mode)
    else:
        print(area1, area2, comparing_contour_mode)

    with (open(pickle_file_path, "wb")) as pickleFile:
        pickle.dump((scan_id, cr.get_volume_data()), pickleFile)
        pickle.dump(result, pickleFile)



in_dir = sys.argv[1]
out_dir = sys.argv[2]

if not os.path.isdir(in_dir):
    logger.error("Invalid input directory: {}".format(in_dir))
else:
    patient_folders = sorted(os.listdir(in_dir))
    for patient_folder in patient_folders:
        create_pickle_for_patient(os.path.join(in_dir, patient_folder), out_dir)
