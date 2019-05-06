from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
from keras.models import Model
from keras.models import load_model
import os
import cv2



def make_dir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    return dir_

def load_inception_v3_model(filename):
    model_name = filename
    model = load_model(model_name)
    model = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d_1').output)

    # print model
    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    return model

def prepare_data(args):

    SZ = args.size
    model = load_inception_v3_model(args.model_name)
    dir_name = "./data"
    fold2 = ["train", "validation", "test"]
    fold3 = ["ASD", "DCM", "HP", "MI", "NORM"]
    save_dir = make_dir(args.save_dir)

    for f1 in fold2:
        for f2 in fold3:
            path2image = os.path.join(os.path.join(dir_name, f1), f2)
            all_images = [x for x in sorted(os.listdir(path2image)) if x[-4:] == '.png']
            SZ = 224
            x_data = np.empty((len(all_images), SZ, SZ, 3), dtype='float32')
            for i, name in enumerate(all_images):
                im = cv2.imread(path2image + name, cv2.IMREAD_COLOR)
                im = cv2.resize(im, dsize=(SZ, SZ), interpolation=cv2.INTER_LANCZOS4)
                im = im / 255.
                x_data[i, ::, ::, ::] = im

            y = model.predict(x_data)
            np.savetxt(os.path.join(save_dir, f1 + "_" + f2 + "_data.txt"), y, delimiter=',')
            with open(os.path.join(save_dir,f1 + "_" + f2 + "_name.txt"), "w") as f:
                for s in all_images:
                    f.write(str(s) + "\n")
            f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="default",
        help="model weights of retrained inception v3"
    )
    parser.add_argument(
        "--save_dir",
        "-s",
        type=str,
        default="default",
        help="filename to save outputs of retrained inception v3"
    )
    parser.add_argument(
        "--size",
        "-i",
        type=int,
        default=224,
        help="How many classes to diagnose"
    )

    args = parser.parse_args()
    prepare_data(args)