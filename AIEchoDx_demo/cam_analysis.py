from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pandas import DataFrame
import argparse
from keras.models import load_model
import cv2
import os
import glob
from keras import backend as K


def make_dir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    return dir_


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def visualize_class_activation_map_v3(model, img_path, output_path, group):
    """
    change group value for different groups

    """

    model = model
    original_img = cv2.imread(img_path, 1)
    # print(img_path)
    width, height, _ = original_img.shape

    img = cv2.resize(original_img, dsize=(224, 224), interpolation=cv2.INTER_LANCZOS4)  #
    img = img / 255.  #
    img = img[np.newaxis, ::, ::, ::]  #

    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = get_output_layer(model, "mixed10")  #
    get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img])
    conv_outputs = conv_outputs[0, :, :, :]

    # Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])  #
    for i, w in enumerate(class_weights[:, group]):  #
        cam += w * conv_outputs[:, :, i]
    print("predictions", predictions)
    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0  # 0.002
    img = heatmap * 0.5 + original_img  # 0.5
    cv2.imwrite(output_path, img)

    return


def load_inception_v3_model(filename):
    model_name = filename
    model = load_model(model_name)

    # print model
    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    return model


def cam_analysis(args):
    # load retrained inception v3 model
    model = load_inception_v3_model(args.modelname)

    # load
    dcm_dir = args.inputdir  # file dir of the echo images
    file_name = [x for x in glob.glob(dcm_dir + "*.png")]
    file_name = DataFrame(data=file_name, columns=["dir_"])
    file_name["patients"] = ""
    for idx, y in enumerate(file_name.dir_):
        y1 = y.split("\\")[-1]
        length = len(y1.split("_")[-1])
        y2 = y1[:-(length + 1)]
        file_name.loc[idx, "patients"] = y2

    for idx, x in enumerate(args.outputdir):

        # ASD:0; DCM:1, HCM:2, pMI:3， NORM:4
        group = args.group
        output_path = os.path.join(args.outputdir, "cam_output")
        make_dir(dir_=output_path)
        dataframe = file_name[file_name["patients"] == x]

        if dataframe.empty:
            print(x)
            print("\t")
        else:
            for idx2, y in enumerate(dataframe["dir_"]):
                img_path = y
                directory = output_path + x
                make_dir(dir_=directory)

                fil = img_path.split("/")[-1]
                output_path2 = directory + "/" + fil
                visualize_class_activation_map_v3(model, img_path, output_path2, group)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modelname",
        "-m",
        type=str,
        default="default",
        help="model weights of retrained inception v3"
    )
    parser.add_argument(
        "--inputdir",
        "-m",
        type=str,
        default="default",
        help="model weights of retrained inception v3"
    )
    parser.add_argument(
        "--outputdir",
        "-o",
        type=str,
        default="default",
        help="filename to save outputs of retrained inception v3"
    )
    parser.add_argument(
        "--group",
        "-g",
        type=int,
        default=1,
        help="How many classes to diagnose"
    )

    args = parser.parse_args()
    cam_analysis(args)
