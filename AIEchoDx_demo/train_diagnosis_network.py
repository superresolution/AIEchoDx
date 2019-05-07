from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import argparse
import pandas as pd
from keras.models import Sequential
from keras import layers
import os
from keras.optimizers import SGD
from keras.callbacks import (
    CSVLogger,
    ModelCheckpoint)


#parameters

LR = 0.0001
FRAMES = 45
classes = ["ASD","DCM","HP","MI","NORM"]

def inception_v3_outputs_to_array(txt_dir, data_dir, frames, clips_no=5):
    """
    channel last as keras

    """

    # read txt file
    txt = pd.read_csv(txt_dir, header=None)
    txt = txt.rename(index=str, columns={0: "slice_name"})
    txt["dir_name"] = ""
    txt["rank"] = ""

    # k = 0
    for i, name in enumerate(txt["slice_name"]):
        name_list = name[:-4].split("_")
        txt["rank"][i] = float(name_list[-1])
        length = 4 + len(name_list[-1]) + 1
        txt["dir_name"][i] = name[:-length]
        # k = k + 1
        # if k>50:
        #    break;

    txt = pd.DataFrame(txt)

    # read data file

    data = pd.read_csv(data_dir, header=None)
    feat_cols = ['pixel_' + str(i) for i in range(data.shape[1])]
    data = pd.DataFrame(data.values, columns=feat_cols)

    data = pd.DataFrame(data)

    # merge data and txt togather

    data2 = data.copy()
    data2["slice_name"] = ""
    data2["dir_name"] = ""
    data2["rank"] = "0"

    for i, value in enumerate(txt.iloc[:, 0]):
        data2.loc[i, "slice_name"] = value
    for i, value in enumerate(txt.iloc[:, 1]):
        data2.loc[i, "dir_name"] = value
    for i, value in enumerate(txt.iloc[:, 2]):
        data2.loc[i, "rank"] = value

        # sort data2 into the right order

    data2 = data2.sort_values(by=['dir_name', 'rank'])
    data2 = data2.reset_index(drop=True)

    # transfer 2048 dimension data to 512*15 data

    sample_name = data2.dir_name.unique()
    length = len(sample_name)
    n_array = np.zeros(shape=(length * clips_no, frames, data.shape[1]))

    count = 0
    for name in data2.dir_name.unique():
        (x, y) = data2[data2["dir_name"] == name].shape
        npy = data2[data2["dir_name"] == name].iloc[:, :data.shape[1]].values
        if x < frames:
            kk = 0
            for cn in range(clips_no):
                for i in range(frames):
                    n_array[count, i, ::] = npy[i % x]
                count = count + 1
        elif (x >= frames) & (x < (frames + clips_no - 1)):
            kk = 1
            for cn in range(clips_no):
                t = cn % (x - frames + 1)
                for i in range(frames):
                    tt = t + i
                    n_array[count, i, ::] = npy[tt]
                count = count + 1
        elif x >= (frames + clips_no - 1):
            kk = 2
            for cn in range(clips_no):
                clips_2 = float(clips_no - 1)
                t = int((x - frames) / clips_2 * cn)
                for i in range(frames):
                    tt = t + i
                    n_array[count, i, ::] = npy[tt]
                count = count + 1
    print(count, length)
    return n_array

def array_concatenate(dir_,FRAMES=45):

    # concatenate arrays generated from ASD, DCM, HCM, pMI and NORM together
    record = {}
    for index, name in enumerate(classes):

        print(name)
        txt_dir = os.path.join(dir_, "train_" + name + "_name.txt")
        data_dir = os.path.join(dir_,"train_" + name + "_data.txt")
        frames = FRAMES
        # clips_no=5
        array = inception_v3_outputs_to_array(txt_dir, data_dir, frames, clips_no=5)

        if array.shape[0] < 400:
            array = np.concatenate((array, array), axis=0)

        record[name] = array.shape
        y_array = np.ones(shape=(array.shape[0], 1))
        y_array = y_array * (index)

        if index == 0:
            X = array.copy()
            y = y_array.copy()
        else:
            X = np.concatenate((X, array), axis=0)
            y = np.concatenate((y, y_array), axis=0)

    return X,y

def build_diagnostic_network(CLASS=5, FRAMES=45):

    # build a diagnostic network with Conv1D layers
    model = Sequential()
    model.add(layers.Conv1D(2048, 2, activation='relu', padding="same", input_shape=(FRAMES, 2048)))
    model.add(layers.Conv1D(512, 2, activation='relu', dilation_rate=2, padding="same"))
    model.add(layers.Flatten())
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dropout(0.5, seed=42))
    model.add(layers.Dense(CLASS, activation='softmax'))

    return model


def train_diagnostic_network(args):
    # parameter
    X_train, y_train = array_concatenate(args.traindir,args.FRAMES)
    X_val, y_val = array_concatenate(args.valdir,args.FRAMES)

    diag_model = build_diagnostic_network(args.CLASS,args.FRAMES)
    diag_model.compile(optimizer=SGD(lr=LR, momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])

    # Checkpoint
    checkpointer = ModelCheckpoint(
        filepath= os.path.join("./model_weights/diagnostic_model","diag_model_{}_checkpoint_{}_{}.h5".format("", "", "")),
        verbose=1,
        save_best_only=True)

    # csvlogger
    csv_logger = CSVLogger(
        os.path.join("./model_weights/diagnostic_model", 'diag_csv_logger_{}_{}_{}.csv'.format("", "", "")))

    diag_model.fit(X_train, y_train,
              epochs=50,
              validation_data=(X_val, y_val),
              callbacks=[csv_logger, checkpointer]
              )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traindir",
        "-t",
        type=str,
        default="default",
        help="dataset folder for training data"
    )
    parser.add_argument(
        "--valdir",
        "-v",
        type=str,
        default="default",
        help="dataset folder for training data"
    )
    parser.add_argument(
        "--FRAMES",
        "-f",
        type=int,
        default=45,
        help="How many classes to diagnose"
    )
    parser.add_argument(
        "--CLASS",
        "-c",
        type=int,
        default=5,
        help="How many classes to diagnose"
    )
    args = parser.parse_args()
    train_diagnostic_network(args)
