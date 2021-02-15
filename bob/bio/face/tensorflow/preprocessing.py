#!/usr/bin/env python
# coding: utf-8

"""
Tensor pre-processing for somr face recognition CNNs
"""

import os
from functools import partial
from multiprocessing import cpu_count
import tensorflow as tf
from tensorflow.keras import layers

# STANDARD FEATURES FROM OUR TF-RECORDS
FEATURES = {
    "data": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "key": tf.io.FixedLenFeature([], tf.string),
}


def decode_tfrecords(x, data_shape, data_type=tf.uint8):
    features = tf.io.parse_single_example(x, FEATURES)
    image = tf.io.decode_raw(features["data"], data_type)
    image = tf.reshape(image, data_shape)
    features["data"] = image
    return features


def get_preprocessor(output_shape):
    """

    """
    preprocessor = tf.keras.Sequential(
        [
            # rotate before cropping
            # 5 random degree rotation
            layers.experimental.preprocessing.RandomRotation(5 / 360),
            layers.experimental.preprocessing.RandomCrop(
                height=output_shape[0], width=output_shape[1]
            ),
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            # FIXED_STANDARDIZATION from https://github.com/davidsandberg/facenet
            # [-0.99609375, 0.99609375]
            # layers.experimental.preprocessing.Rescaling(
            #    scale=1 / 128, offset=-127.5 / 128
            # ),
            layers.experimental.preprocessing.Rescaling(scale=1 / 255, offset=0),
        ]
    )
    return preprocessor


def preprocess(preprocessor, features, augment=False):
    image = features["data"]
    label = features["label"]
    image = preprocessor(image, training=augment)
    return image, label


def prepare_dataset(
    tf_record_paths,
    batch_size,
    epochs,
    data_shape,
    output_shape,
    shuffle=False,
    augment=False,
    autotune=tf.data.experimental.AUTOTUNE,
    n_cpus=cpu_count(),
    shuffle_buffer=int(2e4),
):
    """
    Create batches from a list of TF-Records

    Parameters
    ----------

    tf_record_paths: list
       List of paths of the TF-Records

    batch_size: int

    epochs: int

    shuffle: bool

    augment: bool

    autotune: int

    n_cpus: int

    shuffle_buffer: int
    """

    ds = tf.data.Dataset.list_files(tf_record_paths, shuffle=shuffle)
    ds = tf.data.TFRecordDataset(ds, num_parallel_reads=n_cpus)
    if shuffle:
        # ignore order and read files as soon as they come in
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        ds = ds.with_options(ignore_order)
    ds = ds.map(partial(decode_tfrecords, data_shape=data_shape)).prefetch(
        buffer_size=autotune
    )
    if shuffle:
        ds = ds.shuffle(shuffle_buffer).repeat(epochs)
    preprocessor = get_preprocessor(output_shape)
    ds = ds.batch(batch_size).map(
        partial(preprocess, preprocessor, augment=augment), num_parallel_calls=autotune,
    )

    # Use buffered prefecting on all datasets
    return ds.prefetch(buffer_size=autotune)
