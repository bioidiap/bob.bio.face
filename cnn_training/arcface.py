#!/usr/bin/env python
# coding: utf-8

"""
Trains some face recognition baselines using ARC based models

# ARCFACE PARAMETERS from eq.4
# FROM https://github.com/deepinsight/insightface/blob/master/recognition/ArcFace/sample_config.py#L153
 M1 = 1.0
 M2 = 0.3
 M3 = 0.2


# ARCFACE PARAMETERS from eq.3
M = 0.5  # ArcFace Margin #CHECK SECTION 3.1
SCALE = 64.0  # Scale
# ORIGINAL = False  # Original implementation


The config file has the following format to train an ARCFACE model:

```yml
# VGG2 params
batch-size: 90
face-size: 182
face-output-size: 160
n-classes: 87662


## Backbone
backbone: 'mobilenet-v2'
head: 'arcface'
s: 10
bottleneck: 512
m: 0.5

# Training parameters
#solver: "rmsprop"
solver: "sgd"
lr: 0.1
dropout-rate: 0.5
epochs: 310
lerning-rate-schedule: 'cosine-decay-restarts'



train-tf-record-path: "/path/*.tfrecord"
validation-tf-record-path: "/path/lfw_pairs.tfrecord"
```




Usage:
    arcface.py <config-yaml> <checkpoint_path> 
    arcface.py -h | --help

Options:
  -h --help             Show this screen.
  arcface.py arcface -h | help

"""

import os
from functools import partial

import pkg_resources
import tensorflow as tf
from bob.learn.tensorflow.models.inception_resnet_v2 import InceptionResNetV2
from bob.learn.tensorflow.metrics import predict_using_tensors
from tensorflow.keras import layers
from bob.learn.tensorflow.callbacks import add_backup_callback
from bob.learn.tensorflow.metrics.embedding_accuracy import accuracy_from_embeddings
from bob.extension import rc
from bob.bio.face.tensorflow.preprocessing import prepare_dataset
import yaml

from bob.learn.tensorflow.layers import (
    add_bottleneck,
    add_top,
    SphereFaceLayer,
    ModifiedSoftMaxLayer,
)

from bob.learn.tensorflow.models import (
    EmbeddingValidation,
    ArcFaceLayer,
    ArcFaceModel,
    ArcFaceLayer3Penalties,
)


##############################
# CNN Backbones
# Add your NN backbone here
##############################
BACKBONES = dict()
BACKBONES["inception-resnet-v2"] = InceptionResNetV2
BACKBONES["efficientnet-B0"] = tf.keras.applications.EfficientNetB0
BACKBONES["resnet50"] = tf.keras.applications.ResNet50
BACKBONES["mobilenet-v2"] = tf.keras.applications.MobileNetV2

##############################
# SOLVER SPECIFICATIONS
##############################

SOLVERS = dict()
# Parameters taken from https://github.com/davidsandberg/facenet/blob/master/src/facenet.py#L181
# Fixing the start learning rate
learning_rate = 0.1
SOLVERS["rmsprop"] = partial(
    tf.keras.optimizers.RMSprop,
    learning_rate=learning_rate,
    rho=0.9,
    momentum=0.9,
    epsilon=1.0,
)
SOLVERS["adam"] = partial(tf.keras.optimizers.Adam, learning_rate=learning_rate)
SOLVERS["adagrad"] = partial(tf.keras.optimizers.Adagrad, learning_rate=learning_rate)
SOLVERS["sgd"] = partial(
    tf.keras.optimizers.SGD, learning_rate=learning_rate, momentum=0.9, nesterov=True
)


################################
# DATA SPECIFICATION
###############################
DATA_SHAPES = dict()

# Inputs with 182x182 are cropped to 160x160
DATA_SHAPES[182] = 160
DATA_SHAPES[112] = 98
DATA_SHAPES[126] = 112


# SHAPES EXPECTED FROM THE DATASET USING THIS BACKBONE
# DATA_SHAPE = (182, 182, 3)  # size of faces
DATA_TYPE = tf.uint8
# OUTPUT_SHAPE = (160, 160)

AUTOTUNE = tf.data.experimental.AUTOTUNE

# HERE WE VALIDATE WITH LFW RUNNING A
# INFORMATION ABOUT THE VALIDATION SET
# VALIDATION_TF_RECORD_PATHS = rc["bob.bio.face.cnn.lfw_tfrecord_path"]

# there are 2812 samples in the validation set
VALIDATION_SAMPLES = 2812
VALIDATION_BATCH_SIZE = 38


def create_model(
    n_classes, model_spec, backbone, bottleneck, dropout_rate, input_shape
):

    if backbone == "inception-resnet-v2":
        pre_model = BACKBONES[backbone](
            include_top=False, bottleneck=False, input_shape=input_shape,
        )
    else:
        pre_model = BACKBONES[backbone](
            include_top=False, input_shape=input_shape, weights=None,
        )

    # Adding the bottleneck
    pre_model = add_bottleneck(
        pre_model, bottleneck_size=bottleneck, dropout_rate=dropout_rate
    )
    pre_model = add_top(pre_model, n_classes=n_classes)

    float32_layer = layers.Activation("linear", dtype="float32")

    embeddings = tf.nn.l2_normalize(
        pre_model.get_layer("embeddings/BatchNorm").output, axis=1
    )

    logits_premodel = float32_layer(pre_model.get_layer("logits").output)

    # Wrapping the embedding validation
    pre_model = EmbeddingValidation(
        pre_model.input, outputs=[logits_premodel, embeddings], name=pre_model.name
    )

    ################################
    ## Creating the specific models
    if "arcface" in model_spec:
        labels = tf.keras.layers.Input([], name="label")
        logits_arcface = ArcFaceLayer(
            n_classes, s=model_spec["arcface"]["s"], m=model_spec["arcface"]["m"]
        )(embeddings, labels)
        arc_model = ArcFaceModel(
            inputs=(pre_model.input, labels), outputs=[logits_arcface, embeddings]
        )
    elif "arcface-3p" in model_spec:
        labels = tf.keras.layers.Input([], name="label")
        logits_arcface = ArcFaceLayer3Penalties(
            n_classes,
            s=model_spec["arcface-3p"]["s"],
            m1=model_spec["arcface-3p"]["m1"],
            m2=model_spec["arcface-3p"]["m2"],
            m3=model_spec["arcface-3p"]["m3"],
        )(embeddings, labels)
        arc_model = ArcFaceModel(
            inputs=(pre_model.input, labels), outputs=[logits_arcface, embeddings]
        )
    elif "sphereface" in model_spec:
        logits_arcface = SphereFaceLayer(n_classes, m=model_spec["sphereface"]["m"],)(
            embeddings
        )
        arc_model = EmbeddingValidation(
            pre_model.input, outputs=[logits_arcface, embeddings]
        )

    elif "modified-softmax" in model_spec:
        logits_modified_softmax = ModifiedSoftMaxLayer(n_classes)(embeddings)
        arc_model = EmbeddingValidation(
            pre_model.input, outputs=[logits_modified_softmax, embeddings]
        )

    return pre_model, arc_model


def build_and_compile_models(
    n_classes, optimizer, model_spec, backbone, bottleneck, dropout_rate, input_shape
):
    pre_model, arc_model = create_model(
        n_classes, model_spec, backbone, bottleneck, dropout_rate, input_shape
    )

    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, name="cross_entropy"
    )

    pre_model.compile(optimizer=optimizer, loss=cross_entropy, metrics=["accuracy"])

    arc_model.compile(optimizer=optimizer, loss=cross_entropy, metrics=["accuracy"])

    return pre_model, arc_model


def train_and_evaluate(
    tf_record_paths,
    checkpoint_path,
    n_classes,
    batch_size,
    epochs,
    model_spec,
    backbone,
    optimizer,
    bottleneck,
    dropout_rate,
    face_size,
    validation_path,
    lerning_rate_schedule,
):

    # number of training steps to do before validating a model. This also defines an epoch
    # for keras which is not really true. We want to evaluate every 180000 (90 * 2000)
    # samples
    # STEPS_PER_EPOCH = 180000 // batch_size
    # KERAS_EPOCH_MULTIPLIER = 6
    STEPS_PER_EPOCH = 2000

    DATA_SHAPE = (face_size, face_size, 3)
    OUTPUT_SHAPE = (DATA_SHAPES[face_size], DATA_SHAPES[face_size])

    if validation_path is None:
        validation_path = rc["bob.bio.face.cnn.lfw_tfrecord_path"]
        if validation_path is None:
            raise ValueError(
                "No validation set was set. Please, do `bob config set bob.bio.face.cnn.lfw_tfrecord_path [PATH]`"
            )

    train_ds = prepare_dataset(
        tf_record_paths,
        batch_size,
        epochs,
        data_shape=DATA_SHAPE,
        output_shape=OUTPUT_SHAPE,
        shuffle=True,
        augment=True,
    )

    val_ds = prepare_dataset(
        validation_path,
        data_shape=DATA_SHAPE,
        output_shape=OUTPUT_SHAPE,
        epochs=epochs,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        augment=False,
    )
    val_metric_name = "val_accuracy"

    pre_model, arc_model = build_and_compile_models(
        n_classes,
        optimizer,
        model_spec,
        backbone,
        bottleneck=bottleneck,
        dropout_rate=dropout_rate,
        input_shape=OUTPUT_SHAPE + (3,),
    )

    def scheduler(epoch, lr):
        # 200 epochs at 0.1, 10 at 0.01 and 5 0.001
        # The epoch number here is Keras's which is different from actual epoch number
        # epoch = epoch // KERAS_EPOCH_MULTIPLIER

        # Tracking in the tensorboard
        tf.summary.scalar("learning rate", data=lr, step=epoch)

        if epoch in range(200):
            return 1 * lr
        else:
            return lr * tf.math.exp(-0.01)

    if lerning_rate_schedule == "cosine-decay-restarts":
        decay_steps = 50
        lr_decayed_fn = tf.keras.callbacks.LearningRateScheduler(
            tf.keras.experimental.CosineDecayRestarts(
                0.1, decay_steps, t_mul=2.0, m_mul=0.8, alpha=0.1
            ),
            verbose=1,
        )

    else:
        lr_decayed_fn = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

    callbacks = {
        "latest": tf.keras.callbacks.ModelCheckpoint(
            f"{checkpoint_path}/latest", verbose=1
        ),
        "tensorboard": tf.keras.callbacks.TensorBoard(
            log_dir=f"{checkpoint_path}/logs", update_freq=15, profile_batch=0
        ),
        "lr": lr_decayed_fn,
        "nan": tf.keras.callbacks.TerminateOnNaN(),
    }

    callbacks = add_backup_callback(callbacks, backup_dir=f"{checkpoint_path}/backup")
    # STEPS_PER_EPOCH
    pre_model.fit(
        train_ds,
        epochs=2,
        validation_data=val_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_SAMPLES // VALIDATION_BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    # STEPS_PER_EPOCH
    # epochs=epochs * KERAS_EPOCH_MULTIPLIER,
    arc_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_SAMPLES // VALIDATION_BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )


from docopt import docopt

if __name__ == "__main__":
    args = docopt(__doc__)

    config = yaml.full_load(open(args["<config-yaml>"]))

    model_spec = dict()
    if config["head"] == "arcface":
        model_spec["arcface"] = dict()
        model_spec["arcface"]["m"] = float(config["m"])
        model_spec["arcface"]["s"] = int(config["s"])

    if config["head"] == "arcface-3p":
        model_spec["arcface-3p"] = dict()
        model_spec["arcface-3p"]["m1"] = float(config["m1"])
        model_spec["arcface-3p"]["m2"] = float(config["m2"])
        model_spec["arcface-3p"]["m3"] = float(config["m3"])
        model_spec["arcface-3p"]["s"] = int(config["s"])

    if config["head"] == "sphereface":
        model_spec["sphereface"] = dict()
        model_spec["sphereface"]["m"] = float(config["m"])

    if config["head"] == "modified-softmax":
        # There's no hyper parameter here
        model_spec["modified-softmax"] = dict()

    train_and_evaluate(
        config["train-tf-record-path"],
        args["<checkpoint_path>"],
        int(config["n-classes"]),
        int(config["batch-size"]),
        int(config["epochs"]),
        model_spec,
        config["backbone"],
        optimizer=SOLVERS[config["solver"]](learning_rate=float(config["lr"])),
        bottleneck=int(config["bottleneck"]),
        dropout_rate=float(config["dropout-rate"]),
        face_size=int(config["face-size"]),
        validation_path=config["validation-tf-record-path"],
        lerning_rate_schedule=config["lerning-rate-schedule"],
    )

