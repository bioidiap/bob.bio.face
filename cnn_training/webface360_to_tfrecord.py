#!/usr/bin/env python
# coding: utf-8

"""
Converts WEB360 to TF RECORD

Usage:
    webface360_to_tfrecords.py <web360-path> <output-path> 
    webface360_to_tfrecords.py -h | --help

Options:
  -h --help             Show this screen.  

"""


from docopt import docopt
import numpy as np
import os
import bob.io.image
import bob.io.base
import tensorflow as tf
import sys
from datetime import datetime


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_web360dirs():
    """
    Here I'm hardcoding the paths so we get consistent tfrecords,
    just in case the IT decides to reestructure the web360 directory
    """

    return [
        "0_0_000",
        "0_0_001",
        "0_0_002",
        "0_1_003",
        "0_1_004",
        "0_1_005",
        "0_2_006",
        "0_2_007",
        "0_2_008",
        "0_3_009",
        "0_3_010",
        "0_3_011",
        "0_4_012",
        "0_4_013",
        "0_4_014",
        "0_5_015",
        "0_5_016",
        "0_5_017",
        "0_6_018",
        "0_6_019",
        "0_6_020",
        "1_0_000",
        "1_0_001",
        "1_0_002",
        "1_1_003",
        "1_1_004",
        "1_1_005",
        "1_2_006",
        "1_2_007",
        "1_2_008",
        "1_3_009",
        "1_3_010",
        "1_3_011",
        "1_4_012",
        "1_4_013",
        "1_4_014",
        "1_5_015",
        "1_5_016",
        "1_5_017",
        "1_6_018",
        "1_6_019",
        "1_6_020",
        "2_0_000",
        "2_0_001",
        "2_0_002",
        "2_1_003",
        "2_1_004",
        "2_1_005",
        "2_2_006",
        "2_2_007",
        "2_2_008",
        "2_3_009",
        "2_3_010",
        "2_3_011",
        "2_4_012",
        "2_4_013",
        "2_4_014",
        "2_5_015",
        "2_5_016",
        "2_5_017",
        "2_6_018",
        "2_6_019",
        "2_6_020",
    ]


def get_keys(base_path):

    root_dirs = get_web360dirs()
    keys = dict()
    offset = 0
    for r in root_dirs:

        identities_dir = os.path.join(base_path, r)
        for i in os.listdir(identities_dir):
            id_dir = os.path.join(identities_dir, i)
            if os.path.isdir(id_dir):
                keys[i] = offset
                offset += 1
    return keys


def generate_tfrecord(
    chunk_path, output_tf_record_path, keys,
):
    def write_single_line_tfrecord(writer, image, offset, user_id):

        # Serializing
        serialized_img = image.tobytes()

        # Writing
        feature = {
            "data": _bytes_feature(serialized_img),
            "label": _int64_feature(offset),
            "key": _bytes_feature(str.encode(user_id)),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    os.makedirs(os.path.dirname(output_tf_record_path), exist_ok=True)

    with tf.io.TFRecordWriter(output_tf_record_path) as tf_writer:

        for identity in os.listdir(chunk_path):
            # Discarting the one we've discarted already
            if identity not in keys:
                continue

            identity_path = os.path.join(chunk_path, identity)
            if not os.path.isdir(identity_path):
                continue

            sys.stdout.write(f"Processing {identity} \n")
            sys.stdout.flush()

            for image_path in os.listdir(identity_path):
                image_path = os.path.join(identity_path, image_path)
                if os.path.splitext(image_path)[-1] != ".jpg":
                    continue
                image = bob.io.image.to_matplotlib(bob.io.image.load(image_path))

                write_single_line_tfrecord(tf_writer, image, keys[identity], identity)


if __name__ == "__main__":
    args = docopt(__doc__)

    WEB360_PATH = args["<web360-path>"]
    output_path = args["<output-path>"]

    if "SGE_TASK_LAST" in os.environ:
        TOTAL_CHUNKS = int(os.environ["SGE_TASK_LAST"])
        CURRENT_CHUNK = int(os.environ["SGE_TASK_ID"]) - 1
    else:
        TOTAL_CHUNKS = 1
        CURRENT_CHUNK = 0

    # keys = get_keys(WEB360_PATH)
    import pickle

    keys = pickle.loads(open("keys-web360.pickle", "rb").read())

    root_dirs = get_web360dirs()
    output_tf_record_path = os.path.join(output_path, f"chunk_{CURRENT_CHUNK}.tfrecord")
    chunk_path = os.path.join(WEB360_PATH, root_dirs[CURRENT_CHUNK])

    generate_tfrecord(chunk_path, output_tf_record_path, keys)

