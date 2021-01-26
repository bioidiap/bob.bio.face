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


Usage:
    arcface.py <tf_record_paths> <checkpoint_path> [--dropout-rate=<kn> --bottleneck=<kn> --lr=<kn> --solver=<kn> --backbone=<kn> --n-classes=<kn> --epochs=<kn> --batch-size=<kn>] arcface [--m=<kn> --s=<kn>]
    arcface.py <tf_record_paths> <checkpoint_path> [--dropout-rate=<kn> --bottleneck=<kn> --lr=<kn> --solver=<kn> --backbone=<kn> --n-classes=<kn> --epochs=<kn> --batch-size=<kn>] arcface-3p [--m1=<kn> --m2=<kn> --m3=<kn> --s=<kn>]
    arcface.py <tf_record_paths> <checkpoint_path> [--dropout-rate=<kn> --bottleneck=<kn> --lr=<kn> --solver=<kn> --backbone=<kn> --n-classes=<kn> --epochs=<kn> --batch-size=<kn>] sphereface [--m=<kn>]
    arcface.py <tf_record_paths> <checkpoint_path> [--dropout-rate=<kn> --bottleneck=<kn> --lr=<kn> --solver=<kn> --backbone=<kn> --n-classes=<kn> --epochs=<kn> --batch-size=<kn>] modified-softmax
    arcface.py -h | --help

Options:
  -h --help             Show this screen.
  --m=<kn>              ArcFace/SphereFace Margin [default: 0.5]
  --m1=<kn>             ArcFace Margin [default: 1.0]
  --m2=<kn>             ArcFace Margin [default: 0.3]
  --m3=<kn>             ArcFace Margin [default: 0.2]
  --s=<kn>              ArcFace Scale [default: 64] 
  --n-classes=<kn>      Number of classes [default: 87662]
  --epochs=<kn>         Epochs [default: 35]
  --lr=<kn>             Learning Rate [default: 0.1]
  --batch-size=<kn>     Batch size [default: 90]
  --backbone=<kn>       DCNN Backbone (Options available are: `inception-resnet-v2`, `efficientnet-B0`, `resnet50` and `mobilenet-v2`) [default: inception-resnet-v2]
  --solver=<kn>         Solver used during training (Options available are: `rmsprop`, `adam`, `adagrad`) [default: adam]
  --bottleneck=<kn>     Bottleneck size [default: 128]
  --dropout-rate=<kn>   Dropout Rate used before the bottleneck [default: 0.2]
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


# CNN Backbone
# Change your NN backbone here
BACKBONES = dict()
BACKBONES["inception-resnet-v2"] = InceptionResNetV2
BACKBONES["efficientnet-B0"] = tf.keras.applications.EfficientNetB0
BACKBONES["resnet50"] = tf.keras.applications.ResNet50
BACKBONES["mobilenet-v2"] = tf.keras.applications.MobileNetV2

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


# SHAPES EXPECTED FROM THE DATASET USING THIS BACKBONE
DATA_SHAPE = (182, 182, 3)  # size of faces
DATA_TYPE = tf.uint8
OUTPUT_SHAPE = (160, 160)

AUTOTUNE = tf.data.experimental.AUTOTUNE

# HERE WE VALIDATE WITH LFW RUNNING A
# INFORMATION ABOUT THE VALIDATION SET
VALIDATION_TF_RECORD_PATHS = rc["bob.bio.face.cnn.lfw_tfrecord_path"]

# there are 2812 samples in the validation set
VALIDATION_SAMPLES = 2812
VALIDATION_BATCH_SIZE = 38


def create_model(n_classes, model_spec, backbone, bottleneck, dropout_rate):

    if backbone == "inception-resnet-v2":
        pre_model = BACKBONES[backbone](
            include_top=False, bottleneck=False, input_shape=OUTPUT_SHAPE + (3,),
        )
    else:
        pre_model = BACKBONES[backbone](
            include_top=False, input_shape=OUTPUT_SHAPE + (3,), weights=None,
        )

    # Adding the bottleneck
    pre_model = add_bottleneck(
        pre_model, bottleneck_size=bottleneck, dropout_rate=dropout_rate
    )
    pre_model = add_top(pre_model, n_classes=n_classes)

    float32_layer = layers.Activation("linear", dtype="float32")

    embeddings = tf.nn.l2_normalize(pre_model.get_layer("embeddings").output, axis=1)

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
    n_classes, optimizer, model_spec, backbone, bottleneck, dropout_rate
):
    pre_model, arc_model = create_model(
        n_classes, model_spec, backbone, bottleneck, dropout_rate
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
):
    # number of training steps to do before validating a model. This also defines an epoch
    # for keras which is not really true. We want to evaluate every 180000 (90 * 2000)
    # samples
    STEPS_PER_EPOCH = 180000 // batch_size
    KERAS_EPOCH_MULTIPLIER = 6

    train_ds = prepare_dataset(
        tf_record_paths,
        batch_size,
        epochs,
        data_shape=DATA_SHAPE,
        output_shape=OUTPUT_SHAPE,
        shuffle=True,
        augment=True,
    )

    if VALIDATION_TF_RECORD_PATHS is None:
        raise ValueError(
            "No validation set was set. Please, do `bob config set bob.bio.face.cnn.lfw_tfrecord_path [PATH]`"
        )

    val_ds = prepare_dataset(
        VALIDATION_TF_RECORD_PATHS,
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
    )

    def scheduler(epoch, lr):
        # 20 epochs at 0.1, 10 at 0.01 and 5 0.001
        # The epoch number here is Keras's which is different from actual epoch number
        epoch = epoch // KERAS_EPOCH_MULTIPLIER

        # Tracking in the tensorboard
        tf.summary.scalar("learning rate", data=lr, step=epoch)

        if epoch in range(20):
            return 0.1
        elif epoch in range(20, 30):
            return 0.01
        else:
            return 0.001

    callbacks = {
        "latest": tf.keras.callbacks.ModelCheckpoint(
            f"{checkpoint_path}/latest", verbose=1
        ),
        "tensorboard": tf.keras.callbacks.TensorBoard(
            log_dir=f"{checkpoint_path}/logs", update_freq=15, profile_batch=0
        ),
        "lr": tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
        "nan": tf.keras.callbacks.TerminateOnNaN(),
    }

    callbacks = add_backup_callback(callbacks, backup_dir=f"{checkpoint_path}/backup")
    # STEPS_PER_EPOCH
    pre_model.fit(
        train_ds,
        epochs=1,
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
        epochs=epochs * KERAS_EPOCH_MULTIPLIER,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_SAMPLES // VALIDATION_BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )


from docopt import docopt

if __name__ == "__main__":
    args = docopt(__doc__)

    model_spec = dict()
    if args["arcface"]:
        model_spec["arcface"] = dict()
        model_spec["arcface"]["m"] = float(args["--m"])
        model_spec["arcface"]["s"] = int(args["--s"])

    if args["arcface-3p"]:
        model_spec["arcface-3p"] = dict()
        model_spec["arcface-3p"]["m1"] = float(args["--m1"])
        model_spec["arcface-3p"]["m2"] = float(args["--m2"])
        model_spec["arcface-3p"]["m3"] = float(args["--m3"])
        model_spec["arcface-3p"]["s"] = int(args["--s"])

    if args["sphereface"]:
        model_spec["sphereface"] = dict()
        model_spec["sphereface"]["m"] = float(args["--m"])

    if args["modified-softmax"]:
        # There's no hyper parameter here
        model_spec["modified-softmax"] = dict()

    train_and_evaluate(
        args["<tf_record_paths>"],
        args["<checkpoint_path>"],
        int(args["--n-classes"]),
        int(args["--batch-size"]),
        int(args["--epochs"]),
        model_spec,
        args["--backbone"],
        optimizer=SOLVERS[args["--solver"]](learning_rate=float(args["--lr"])),
        bottleneck=int(args["--bottleneck"]),
        dropout_rate=float(args["--dropout-rate"]),
    )

