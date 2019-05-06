from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import layers
from keras.optimizers import SGD
from keras.callbacks import (
    CSVLogger,
    ModelCheckpoint)

# parameter
BZ = 500
EPOCH = 50
LR = 0.001
SZ = 224  # or 299 is also OK
DO = 0.5
CLASS = 5  # ASD, DCM, HCM, pMI and NORM

def build_inception_v3_model():
    # create the base pre-trained model
    base_model = InceptionV3(weights="imagenet", input_shape=(SZ, SZ, 3), include_top=False)

    x = base_model.output
    x = layers.Dropout(DO)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DO)(x)
    predictions = layers.Dense(CLASS, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

def train(args):

    classes = ["ASD", "DCM", "HP", "MI", "NORM"]
    my_class_weight = {0: 1., 1: 2., 2: 1., 3: 2., 4: 1., 5: 1.}
    echo_train = os.path.join(args.data_dir,"train")
    echo_validation = os.path.join(args.data_dir,"validation")

    # load model and set model compile:
    model = build_inception_v3_model()
    model.compile(optimizer=SGD(lr=LR, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    # Checkpoint
    checkpointer = ModelCheckpoint(
        filepath="model_checkpoint_{}_{}.h5".format("first", "title"),
        verbose=1,
        save_best_only=True)

    # csvlogger
    csv_logger = CSVLogger(
        'csv_logger_{}_{}.csv'.format("first", "title"))

    # image data generator:
    train_datagen = ImageDataGenerator(
        rotation_range=15.,  # rotation
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1. / 255,
        shear_range=0.,
        zoom_range=0.1,
        channel_shift_range=0.,
        fill_mode='nearest',
        # fill_mode = "constant",
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        preprocessing_function=None)

    train_generator = train_datagen.flow_from_directory(
        echo_train,
        target_size=(SZ, SZ),
        batch_size=BZ,
        shuffle=True,
        classes=classes,
        class_mode='categorical',
        seed=42)

    val_datagen = ImageDataGenerator(
        rescale=1. / 255)

    val_generator = val_datagen.flow_from_directory(
        echo_validation,
        target_size=(SZ, SZ),
        batch_size=BZ,
        shuffle=True,
        classes=classes,
        class_mode='categorical',
        seed=42)

    # train 50 epochs
    model.fit_generator(train_generator,
                        steps_per_epoch=100000. / BZ,
                        epochs=EPOCH,
                        validation_data=val_generator,
                        validation_steps=10000. / BZ,
                        class_weight=my_class_weight,
                        callbacks=[csv_logger, checkpointer])

    # return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        "-d",
        type=str,
        default="default",
        help="folder address of train,validation and test datasets"
    )
    args = parser.parse_args()
    train(args)