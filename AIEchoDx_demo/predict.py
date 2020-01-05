from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from keras.models import Model, load_model
import argparse
from util import load_avi_video, load_dcm_video, remove_info2, limited_equalize

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

    path1 = os.path.join(os.path.abspath('.'), "model_weights/inception_v3_model")
    path2 = os.path.join(os.path.abspath('.'), "model_weights/diagnostic_v3_model")
    inception_v3_model_filename = os.path.join(path1, args.incepname)
    diagnostic_model_filename = os.path.join(path2, args.diagname)

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