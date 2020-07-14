from con_reader import CONreaderVM
from dicom_reader import DCMreaderVM
from utils import get_logger
from domain.patient_data import PatientData
import numpy as np
import pickle
import os
import sys
import cv2 as cv

logger = get_logger(__name__)

def create_path_for_file(pickle_file_path):
    os.makedirs(os.path.dirname(pickle_file_path), exist_ok=True)

def collect_contour_slices_by_frames(contours):
    frameSliceDict = {}
    for slc in contours:
        for frm in contours[slc]:
            if not(frm in frameSliceDict):
                frameSliceDict[frm] = []
            frameSliceDict[frm].append(slc)
    return frameSliceDict

def left_ventricle_contours(contours):
    left_ventricle_color_modes = {"ln", "lp"}
    left_ventricle_contours = {}
    for slc, frames in contours.items():
        for frm, modes in frames.items():
            filtered_contours = dict(filter(lambda contour: contour[0] in left_ventricle_color_modes, modes.items()))
            if len(filtered_contours) == 0:
                continue
            if not(slc in left_ventricle_contours):
                left_ventricle_contours[slc] = {}
            left_ventricle_contours[slc][frm] = filtered_contours
    return left_ventricle_contours

def frame_of_diastole(frame_slice_dict, contours):
    frame1 = list(frame_slice_dict.keys())[0]
    frame2 = list(frame_slice_dict.keys())[1]
    slice_dict_1 = list(frame_slice_dict.values())[0]
    slice_dict_2 = list(frame_slice_dict.values())[1]

    slice_intersection = list(set(slice_dict_1).intersection(set(slice_dict_2)))
    slice_intersection.sort()
    mid_slice_index = slice_intersection[len(slice_intersection)//2]

    common_contour_mode = next(iter(set(contours[mid_slice_index][frame1].keys()).intersection(contours[mid_slice_index][frame2])))

    area1 = cv.contourArea(contours[mid_slice_index][frame1][common_contour_mode].astype(int))
    area2 = cv.contourArea(contours[mid_slice_index][frame2][common_contour_mode].astype(int))
    return frame1 if area1 > area2 else frame2

def calculate_sampling_slices(frame_slice_dict, diastole_frame):
    diastole_slice_indexes = frame_slice_dict[diastole_frame]
    return np.percentile(np.array(diastole_slice_indexes), (19,50,83), interpolation='lower')

def create_contour_diff_matricies(sampling_contours, shape):
    contour_diff_matricies = []
    for contours in sampling_contours:
        contour_diff_mx = np.zeros(shape)
        cv.drawContours(contour_diff_mx, [contours["lp"].astype(np.int32)],0, color=255, thickness=-1)
        cv.drawContours(contour_diff_mx, [contours["ln"].astype(np.int32)],0, color=0, thickness=-1)
        contour_diff_mx = cv.resize(contour_diff_mx, (224,224), interpolation = cv.INTER_AREA)
        contour_diff_matricies.append(contour_diff_mx.astype('uint8'))
    return contour_diff_matricies

def read_pathology(meta_txt):
    pathology = ""
    with open(meta_txt, "r") as f:
        pathology = f.readline().split(": ")[1]
    return pathology.rstrip()

def create_pickle_for_patient(in_dir, out_dir):
    scan_id = os.path.basename(in_dir)
    image_folder = os.path.join(in_dir, "sa", "images")
    con_file = os.path.join(in_dir, "sa", "contours.con")
    meta_txt = os.path.join(in_dir, "meta.txt")

    if not os.path.isdir(image_folder):
        logger.error("Could not find image folder for: {}".format(scan_id))
        return
    if not os.path.isfile(con_file):
        logger.error("Could not find .con file for: {}".format(scan_id))
        return
    if not os.path.isfile(meta_txt):
        logger.error("Could not find meta.txt file for: {}".format(scan_id))
        return

    dr = DCMreaderVM(image_folder)
    if dr.num_frames == 0 and dr.num_frames == 0 or dr.broken:
        logger.error("Could not create pickle file for {}".format(scan_id))
        return

    cr = CONreaderVM(con_file)
    contours = left_ventricle_contours(cr.get_hierarchical_contours())

    frame_slice_dict = collect_contour_slices_by_frames(contours)
    if not (len(frame_slice_dict) == 2):
        logger.error("Too many contour frames for {}".format(scan_id))
        return

    pickle_file_path = os.path.join(out_dir, scan_id + ".p")
    create_path_for_file(pickle_file_path)

    diastole_frame = frame_of_diastole(frame_slice_dict, contours)
    sampling_slices = calculate_sampling_slices(frame_slice_dict, diastole_frame)
    sampling_contours = []
    for slice_index in sampling_slices:
        shape = dr.get_image(slice_index,diastole_frame).shape
        sampling_contours.append(contours[slice_index][diastole_frame])
    pathology = read_pathology(meta_txt)
    shape = dr.get_image(sampling_slices[0],diastole_frame).shape
    contour_diff_matricies = create_contour_diff_matricies(sampling_contours, shape)
    patient_data = PatientData(scan_id, pathology, cr.get_volume_data(), contour_diff_matricies)

    with (open(pickle_file_path, "wb")) as pickleFile:
        pickle.dump(patient_data, pickleFile)

in_dir = sys.argv[1]
out_dir = sys.argv[2]

if not os.path.isdir(in_dir):
    logger.error("Invalid input directory: {}".format(in_dir))
else:
    patient_folders = sorted(os.listdir(in_dir))
    for patient_folder in patient_folders:
        create_pickle_for_patient(os.path.join(in_dir, patient_folder), out_dir)
