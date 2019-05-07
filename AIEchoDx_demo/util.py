import SimpleITK as sitk
from PIL import Image
import numpy as np
import cv2

def load_dcm_video(filename):
    #
    ds = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(ds)
    frame_num, width, height, channel = img_array.shape
    return img_array, frame_num, width, height, channel

def load_avi_video(videoname):
    # load avi video using cv2 package
    cap = cv2.VideoCapture(videoname)
    framenum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    channel = 3

    img_array = np.empty((framenum, height, width, channel), np.dtype('uint8'))
    fc = 0
    ret = True
    while (fc < framenum and ret):
        ret, img_array[fc] = cap.read()
        fc += 1
    cap.release()

    return img_array, frame_num, width, height, channel

def load_dcm_information(filename):
    #load information of
    info = {}
    ds = dicom.read_file(filename)
    info['PatientID'] = ds.PatientID
    info['PatientName'] = ds.PatientName
    info['PatientBirthDate'] = ds.PatientBirthDate
    info['PatientSex'] = ds.PatientSex
    info['StudyID'] = ds.StudyID
    info['StudyDate'] = ds.StudyDate
    info['StudyTime'] = ds.StudyTime
    info['InstitutionName'] = ds.InstitutionName
    info['Manufacturer'] = ds.Manufacturer
    info['NumberOfFrames'] = ds.NumberOfFrames
    return info

def show_image(img_array, frame_num = 0):
    img_bitmap = Image.fromarray(img_array[frame_num])
    return img_bitmap

def limited_equalize(img_array, limit = 4.0):
    img_array_list = []
    for img in img_array:
        clahe = cv2.createCLAHE(clipLimit = limit, tileGridSize = (8,8))
        img_array_list.append(clahe.apply(img))
    img_array_limited_equalized = np.array(img_array_list)
    return img_array_limited_equalized

def remove_info(video):
    #remove periphery information
    l, row, col = video.shape[:3]
    for i in range(row):
        for j in range(col):
            if (video[:,i,j,0] == np.array([video[0,i,j,0]] * l)).all():
                video[:,i,j] = np.zeros((l,3))
    return

def remove_info2(video):
    #remove periphery inforvideomation
    l, row, col = video.shape[:3]
    video2 = video.copy()
    for i in range(row):
        for j in range(col):
            if abs(video[:,i,j,0].astype("float32") - np.array([video[0,i,j,0]] * l).astype("float32")).max()<10:
                video2[:,i,j] = np.zeros((l,3))
    return video2



def write_video(img_array):
    frame_num, width, height = img_array.shape
    filename_output = filename.split('.')[0] + '.avi'
    video = cv2.VideoWriter(filename_output, -1, 16, (width, height))
    for img in img_array:
        video.write(img)
    video.release()