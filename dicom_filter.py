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

def calculate_contour_slice_window(fame_slice_dict):
    start_slice = max([min(slices) for frame, slices in fame_slice_dict.items()])
    end_slice = min([max(slices) for frame, slices in fame_slice_dict.items()])
    return (start_slice, end_slice)

def swap_phases(patient_data):
    comparing_contour_mode = common_contour_mode(patient_data)
    systole_area = cv.contourArea(patient_data.mid_systole_contours()[comparing_contour_mode].astype(int))
    diastole_area = cv.contourArea(patient_data.mid_diastole_contours()[comparing_contour_mode].astype(int))
    return diastole_area < systole_area

def common_contour_mode(patient_data):
    systole_contours = set(patient_data.mid_systole_contours().keys())
    diastole_contours = patient_data.mid_diastole_contours().keys()
    common_contours = systole_contours.intersection(diastole_contours)
    return next(iter(common_contours))

def calculate_sampling_slices(frame_slice_dict):
    (start_slice, end_slice) = calculate_contour_slice_window(frame_slice_dict)
    window_slices = list(filter(lambda slc : slc>=start_slice and slc<=end_slice, next(iter(frame_slice_dict.values()))))
    return np.percentile(np.array(window_slices), (19,50,83), interpolation='lower')

def add_contour_diff_mx_to_diastole(patient_data):
    diastole_data = patient_data.diastole
    diastole_data_with_contour_diff_mx = map(lambda e : (e[0],e[1],e[2], create_contour_diff_mx(e[1], e[0].shape)), diastole_data)
    return list(diastole_data_with_contour_diff_mx)


def create_contour_diff_mx(contours, shape):
    contour_diff_mx = np.zeros(shape)
    cv.drawContours(contour_diff_mx, [contours["lp"].astype(np.int32)],0, color=255, thickness=-1)
    cv.drawContours(contour_diff_mx, [contours["ln"].astype(np.int32)],0, color=0, thickness=-1)
    contour_diff_mx = cv.resize(contour_diff_mx, (200,200), interpolation = cv.INTER_AREA)
    return contour_diff_mx

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

    sampling_slices = calculate_sampling_slices(frame_slice_dict)
    result = []
    for frm in frame_slice_dict:
        phase = []
        for slc in sampling_slices:
            image = dr.get_image(slc,frm)
            phase.append((image.astype('uint8'), contours[slc][frm], (slc,frm)))
           
        result.append(phase)

    pathology = read_pathology(meta_txt)
    patient_data = PatientData(scan_id, pathology, cr.get_volume_data(), result[0], result[1])
    
    if swap_phases(patient_data):
        patient_data.systole, patient_data.diastole = patient_data.diastole, patient_data.systole

    patient_data.diastole = add_contour_diff_mx_to_diastole(patient_data)

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
