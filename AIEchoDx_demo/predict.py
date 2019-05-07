from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import SimpleITK as SITK
import numpy as np
import cv2
import os
from keras.models import Model, load_model
import argparse

def load_dcm_video(videoname):
    # load dcm video
    d = SITK.ReadImage(videoname)
    img_array = SITK.GetArrayFromImage(d)
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

    return img_array, framenum, width, height, channel

def remove_info(video):
    #remove periphery information
    l, row, col = video.shape[:3]
    for i in range(row):
        for j in range(col):
            if (video[:,i,j,0] == np.array([video[0,i,j,0]] * l)).all():
                video[:,i,j] = np.zeros((l,3))
    return video

def remove_info2(video):
    #remove periphery information
    l, row, col = video.shape[:3]
    for i in range(row):
        for j in range(col):
            if abs(video[:,i,j,0].astype("float32") - np.array([video[0,i,j,0]] * l).astype("float32")).max()<5: # no larger than 10 were suitable as a threshold
                video[:,i,j] = np.zeros((l,3))
    return video

def limited_equalize(img_array, limit = 4.0):
    img_array_list = []
    for img in img_array:
        clahe = cv2.createCLAHE(clipLimit = limit, tileGridSize = (8,8))
        img_array_list.append(clahe.apply(img))
    img_array_limited_equalized = np.array(img_array_list)
    return img_array_limited_equalized

def predict(args):
    # predict probability
    name, video_extension = os.path.splitext(args.videoname)

    if video_extension == ".DCM":
        img_array, frame_num, width, height, channel = load_dcm_video(args.videoname)
        img_array = remove_info2(img_array)
        echo_video = limited_equalize(img_array)
    elif (video_extension == ".avi") or (video_extension == ".AVI"):
        img_array, frame_num, width, height, channel = load_avi_video(args.videoname)
        img_array = remove_info2(img_array)
        echo_video = limited_equalize(img_array)
    else:
        print("Error video types")

    inception_v3_model_filename = os.path.join("./model_weights/incpetion_v3_model",args.incepname)
    diagnostic_model_filename = os.path.join("./model_weights/diagnostic_v3_model",args.diagname)

    inception_model = load_model(inception_v3_model_filename)
    inception_model_GAP = Model(inputs=inception_model.input,
                            outputs=inception_model.get_layer('global_average_pooling2d_1').output)
    diagnostic_model = load_model(diagnostic_model_filename)

    array = inception_model_GAP.predict(echo_video)
    probs = diagnostic_model.predict(array)

    return probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videoname",
        "-v",
        type=str,
        default="default",
        help="dataset folder for training data"
    )
    parser.add_argument(
        "--incepname",
        "-i",
        type=str,
        default="default",
        help="dataset folder for training data"
    )
    parser.add_argument(
        "--diagname",
        "-d",
        type=str,
        default="default",
        help="dataset folder for training data"
    )
    parser.add_argument(
        "--maxx",
        "-x",
        type=int,
        default=45,
        help="How many classes to diagnose"
    )
    parser.add_argument(
        "--minn",
        "-n",
        type=int,
        default=0,
        help="How many classes to diagnose"
    )
    args = parser.parse_args()
    predict(args)
