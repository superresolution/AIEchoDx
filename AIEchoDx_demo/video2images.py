from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import SimpleITK as sitk
from PIL import Image
import numpy as np
import cv2
import pydicom

# parameter
classes = ["ASD", "DCM", "HCM", "pMI", "NORM"]
(a,b,c) = (3,1,1)
path_in = os.path.join(os.path.abspath('.'), "video_examples")
path_in_list = os.listdir(path_in)
path_out_train = os.path.join(os.path.abspath('.'), "data/train")
path_out_validation = os.path.join(os.path.abspath('.'), "data/validation")
path_out_test = os.path.join(os.path.abspath('.'), "data/test")

#
def make_dir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    return dir_


def load_dcm_video_new(videoname):
    # load dcm video
    d = SITK.ReadImage(videoname)
    video_array = SITK.GetArrayFromImage(d)
    if len(video.shape) == 3:
        frame_num, height, width = video_array.shape
        channel = 1
        img_array = np.empty((frame_num, height, width, 3), np.dtype('uint8'))

        for i in range(frame_num):
            img_array[i, ::, ::, 0] = video_array[i, ::, ::]
            img_array[i, ::, ::, 1] = video_array[i, ::, ::]
            img_array[i, ::, ::, 2] = video_array[i, ::, ::]


    elif len(video.shape) == 4:
        frame_num, height, width, channel = video_array.shape
        img_array = np.empty((frame_num, height, width, channel), np.dtype('uint8'))

        for i in range(frame_num):
            img_array[i, ::, ::, 0] = video_array[i, ::, ::, 0]
            img_array[i, ::, ::, 1] = video_array[i, ::, ::, 0]
            img_array[i, ::, ::, 2] = video_array[i, ::, ::, 0]

    else:
        print("Video shape error:", videoname)
    channel_new = 3
    return img_array, frame_num, height, width, channel_new


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

    return img_array, framenum, width, height, channel


def load_file_information(filename):
    # load information of
    info = {}
    ds = pydicom.read_file(filename)
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


def show_image(img_array, frame_num=0):
    img_bitmap = Image.fromarray(img_array[frame_num])
    return img_bitmap


def limited_equalize(img_array, limit=4.0):
    img_array_list = []
    for img in img_array:
        clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8, 8))
        img_array_list.append(clahe.apply(img))
    img_array_limited_equalized = np.array(img_array_list)
    return img_array_limited_equalized


def remove_info(video):
    # remove periphery information
    l, row, col = video.shape[:3]
    for i in range(row):
        for j in range(col):
            if (video[:, i, j, 0] == np.array([video[0, i, j, 0]] * l)).all():
                video[:, i, j] = np.zeros((l, 3))
    return


def remove_info2(video):
    # remove periphery inforvideomation
    l, row, col = video.shape[:3]
    video2 = video.copy()
    for i in range(row):
        for j in range(col):
            if abs(video[:, i, j, 0].astype("float32") - np.array([video[0, i, j, 0]] * l).astype(
                    "float32")).max() < 10:
                video2[:, i, j] = np.zeros((l, 3))
    return video2


def remove_info3(video, channel=0):
    """
    cv2 package;numpy package
    """

    l, row, col = video.shape[:3]
    array = np.zeros((l, row, col, 3), dtype=int)
    for i in range(l):
        img = video[1, ::, ::, ::]
        mask = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)[1][:, :, channel]
        dst = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
        array[i, ::, ::, ::] = dst
    return array


def remove_info4(video):
    # remove periphery information
    l, row, col = video.shape[:3]
    s = len(video) // 2
    mask = np.sum(np.sum(video[:s], axis=0) - np.sum(video[s:2 * s], axis=0), axis=-1)
    video = np.transpose(video, (1, 2, 3, 0))
    video[mask == 0] = np.zeros((3, l))
    video = np.transpose(video, (3, 0, 1, 2))
    return (video)

def main():

    for class_name in classes:
        video_list = [f for f in path_in_list if class_name in f]
        for i in range(len(video_list)):
            video_path = os.path.join(path_in, video_list[i])
            img_array, framenum, width, height, channel = load_avi_video(video_path)
            img_array = remove_info4(img_array)

            if i < a:
                sub_dir = os.path.join(path_out_train, class_name)
                for j in range(len(img_array)):
                    png_file = os.path.join(sub_dir, video_list[i].split(".")[0] + "-" + str(j) + ".png")
                    cv2.imwrite(png_file, limited_equalize(img_array[j]))
            elif i < a + b:
                sub_dir = os.path.join(path_out_validation, class_name)
                for j in range(len(img_array)):
                    png_file = os.path.join(sub_dir, video_list[i].split(".")[0] + "-" + str(j) + ".png")
                    cv2.imwrite(png_file, limited_equalize(img_array[j]))
            else:
                sub_dir = os.path.join(path_out_test, class_name)
                for j in range(len(img_array)):
                    png_file = os.path.join(sub_dir, video_list[i].split(".")[0] + "-" + str(j) + ".png")
                    cv2.imwrite(png_file, limited_equalize(img_array[j]))

if __name__ == '__main__':
    main()