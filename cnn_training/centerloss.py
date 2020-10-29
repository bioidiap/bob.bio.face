#!/usr/bin/env python
# coding: utf-8

"""
Trains a face recognition CNN using the strategy from the paper

"A Discriminative Feature Learning Approach
for Deep Face Recognition" https://ydwen.github.io/papers/WenECCV16.pdf

The default backbone is the InceptionResnetv2

Do `./bin/python centerloss.py --help` for more information

"""

import os
from functools import partial
import click
import pkg_resources
import tensorflow as tf
from bob.learn.tensorflow.losses import CenterLoss, CenterLossLayer
from bob.learn.tensorflow.models.inception_resnet_v2 import InceptionResNetV2
from bob.learn.tensorflow.metrics import predict_using_tensors
from tensorflow.keras import layers
from bob.learn.tensorflow.callbacks import add_backup_callback
from bob.learn.tensorflow.metrics.embedding_accuracy import accuracy_from_embeddings
from bob.extension import rc
from bob.bio.face.tensorflow.preprocessing import prepare_dataset

# CNN Backbone
# Change your NN backbone here
BACKBONE = InceptionResNetV2

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

# WEIGHTS BEWTWEEN the two losses
LOSS_WEIGHTS = {"cross_entropy": 1.0, "center_loss": 0.01}


class CenterLossModel(tf.keras.Model):
    def compile(
        self,
        cross_entropy,
        center_loss,
        loss_weights,
        train_loss,
        train_cross_entropy,
        train_center_loss,
        test_acc,
        **kwargs,
    ):
        super().compile(**kwargs)
        self.cross_entropy = cross_entropy
        self.center_loss = center_loss
        self.loss_weights = loss_weights
        self.train_loss = train_loss
        self.train_cross_entropy = train_cross_entropy
        self.train_center_loss = train_center_loss
        self.test_acc = test_acc

    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            logits, prelogits = self(images, training=True)
            loss_cross = self.cross_entropy(labels, logits)
            loss_center = self.center_loss(labels, prelogits)
            loss = (
                loss_cross * self.loss_weights[self.cross_entropy.name]
                + loss_center * self.loss_weights[self.center_loss.name]
            )
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.train_loss(loss)
        self.train_cross_entropy(loss_cross)
        self.train_center_loss(loss_center)
        return {
            m.name: m.result()
            for m in [self.train_loss, self.train_cross_entropy, self.train_center_loss]
        }

    def test_step(self, data):
        images, labels = data
        logits, prelogits = self(images, training=False)
        self.test_acc(accuracy_from_embeddings(labels, prelogits))
        return {m.name: m.result() for m in [self.test_acc]}


def create_model(n_classes):

    model = BACKBONE(
        include_top=True,
        classes=n_classes,
        bottleneck=True,
        input_shape=OUTPUT_SHAPE + (3,),
    )

    prelogits = model.get_layer("Bottleneck/BatchNorm").output
    prelogits = CenterLossLayer(
        n_classes=n_classes, n_features=prelogits.shape[-1], name="centers"
    )(prelogits)

    logits = model.get_layer("logits").output
    model = CenterLossModel(
        inputs=model.input, outputs=[logits, prelogits], name=model.name
    )
    return model


def build_and_compile_model(n_classes, learning_rate):
    model = create_model(n_classes)

    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, name="cross_entropy"
    )
    center_loss = CenterLoss(
        centers_layer=model.get_layer("centers"),
        alpha=0.9,
        name="center_loss",
    )

    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=learning_rate, rho=0.9, momentum=0.9, epsilon=1.0
    )

    train_loss = tf.keras.metrics.Mean(name="loss")
    train_cross_entropy = tf.keras.metrics.Mean(name="cross_entropy")
    train_center_loss = tf.keras.metrics.Mean(name="center_loss")

    test_acc = tf.keras.metrics.Mean(name="accuracy")

    model.compile(
        optimizer=optimizer,
        cross_entropy=cross_entropy,
        center_loss=center_loss,
        loss_weights=LOSS_WEIGHTS,
        train_loss=train_loss,
        train_cross_entropy=train_cross_entropy,
        train_center_loss=train_center_loss,
        test_acc=test_acc,
    )
    return model


@click.command()
@click.argument("tf-record-paths")
@click.argument("checkpoint-path")
@click.option(
    "-n",
    "--n-classes",
    default=87662,
    help="Number of classes in the classification problem. Default to `87662`, which is the number of identities in our pruned MSCeleb",
)
@click.option(
    "-b",
    "--batch-size",
    default=90,
    help="Batch size. Be aware that we are using single precision. Batch size should be high.",
)
@click.option(
    "-e", "--epochs", default=35, help="Number of epochs",
)
def train_and_evaluate(tf_record_paths, checkpoint_path, n_classes, batch_size, epochs):
    # number of training steps to do before validating a model. This also defines an epoch
    # for keras which is not really true. We want to evaluate every 180000 (90 * 2000)
    # samples
    STEPS_PER_EPOCH = 180000 // batch_size
    learning_rate = 0.1
    KERAS_EPOCH_MULTIPLIER = 6
    train_ds = prepare_dataset(
        tf_record_paths, 
        batch_size,
        epochs,
        data_shape=DATA_SHAPE,
        output_shape=OUTPUT_SHAPE, 
        shuffle=True,
        augment=True
    )

    if VALIDATION_TF_RECORD_PATHS is None:
        raise ValueError("No validation set was set. Please, do `bob config set bob.bio.face.cnn.lfw_tfrecord_path [PATH]`")

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

    model = build_and_compile_model(n_classes, learning_rate)


    def scheduler(epoch, lr):
        # 20 epochs at 0.1, 10 at 0.01 and 5 0.001
        # The epoch number here is Keras's which is different from actual epoch number
        epoch = epoch // KERAS_EPOCH_MULTIPLIER
        if epoch in range(20):
            return 0.1
        elif epoch in range(20, 30):
            return 0.01
        else:
            return 0.001

    callbacks = {
        "latest": tf.keras.callbacks.ModelCheckpoint(f"{checkpoint_path}/latest", verbose=1),
        "best": tf.keras.callbacks.ModelCheckpoint(
            f"{checkpoint_path}/best",
            monitor=val_metric_name,
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        "tensorboard": tf.keras.callbacks.TensorBoard(
            log_dir=f"{checkpoint_path}/logs", update_freq=15, profile_batch="10,50"
        ),
        "lr": tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
        "nan":tf.keras.callbacks.TerminateOnNaN(),
    }
    callbacks = add_backup_callback(callbacks, backup_dir=f"{checkpoint_path}/backup")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs * KERAS_EPOCH_MULTIPLIER,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_SAMPLES // VALIDATION_BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )


if __name__ == "__main__":
    train_and_evaluate()
