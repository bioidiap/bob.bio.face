#!/usr/bin/env python
# coding: utf-8

"""
Trains some face recognition baselines using ARC based models

Usage:
    vgg2_2_tfrecords.py <vgg-path> <output-path> 
    vgg2_2_tfrecords.py -h | --help

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


def search_landmark(landmark_path, img_path):
    with open(landmark_path) as f:
        next(f)
        for line in f:
            line = line.split(",")
            if img_path in line[0]:
                return np.array(
                    [[float(line[i + 1]), float(line[i + 2])] for i in [0, 2, 4, 6, 8]]
                )
        else:
            return None


from bob.bio.face.preprocessor import FaceCrop


def align(image, annotations, cropped_image_size=(126, 126)):

    cropped_image_height, cropped_image_width = cropped_image_size

    # RIGHT_EYE_POS = (40, 46)
    # LEFT_EYE_POS = (40, 80)
    # cropped_positions = {"leye": LEFT_EYE_POS, "reye": RIGHT_EYE_POS}
    # cropped_positions = {"leye": (49, 72), "reye": (49, 38)}
    cropped_positions = {"leye": (55, 81), "reye": (55, 42)}

    cropper = FaceCrop(
        cropped_image_size=cropped_image_size,
        cropped_positions=cropped_positions,
        color_channel="rgb",
        fixed_positions=None,
        annotator=None,
    )
    return bob.io.image.to_matplotlib(
        cropper.transform([image], [annotations])[0].astype("uint8")
    )


def get_id_by_line(line):
    return line.split("/")[0]


def generate_tfrecord(
    base_path, landmark_path, file_list, output_tf_record_path, indexes
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

    with tf.io.TFRecordWriter(output_tf_record_path) as tf_writer:

        current_id = None
        with open(file_list) as f:
            for file_name in f.readlines():

                user_id = get_id_by_line(file_name)
                if user_id in indexes:

                    img = bob.io.base.load(
                        os.path.join(base_path, file_name).rstrip("\n")
                    )
                    l_name = file_name.rstrip(".jpg\n")

                    if current_id != user_id:
                        current_id = user_id
                        sys.stdout.write(
                            f"Writing user {current_id}. {str(datetime.now())} \n"
                        )
                        sys.stdout.flush()

                    landmarks = search_landmark(landmark_path, l_name)
                    if landmarks[0][0] > landmarks[1][0]:
                        annotations = {
                            "reye": (landmarks[1][1], landmarks[1][0]),
                            "leye": (landmarks[0][1], landmarks[0][0]),
                        }
                    else:
                        annotations = {
                            "reye": (landmarks[0][1], landmarks[0][0]),
                            "leye": (landmarks[1][1], landmarks[1][0]),
                        }
                    if landmarks is None:
                        raise ValueError(f"Landmark for {file_name} not found!")

                    aligned_image = align(img, annotations)
                    write_single_line_tfrecord(
                        tf_writer, aligned_image, int(indexes[user_id]), user_id
                    )


def map_indexes(image_path, n_chunks):
    """
    Create a dictionary mapping the ID to VGG2-ID, like:

    {0: 'n000001'],
    1: 'n000002']}

    """

    indexes = sorted(list(set([l.split("/")[0] for l in open(image_path).readlines()])))

    identities_map = {indexes[i]: i for i in range(len(indexes))}

    # SPLIT THE DICTIONARY IN TOTAL_CHUNKS
    indexes_as_list = list(identities_map.items())
    dict_as_list = np.array_split(indexes_as_list, n_chunks)
    dicts = [dict(d) for d in dict_as_list]

    return dicts


if __name__ == "__main__":
    args = docopt(__doc__)

    VGG2_PATH = args["<vgg-path>"]
    LANDMARK_PATH = os.path.join(VGG2_PATH, "bb_landmark", "loose_landmark_train.csv")

    if "SGE_TASK_LAST" in os.environ:
        TOTAL_CHUNKS = int(os.environ["SGE_TASK_LAST"])
        CURRENT_CHUNK = int(os.environ["SGE_TASK_ID"]) - 1
    else:
        TOTAL_CHUNKS = 1
        CURRENT_CHUNK = 0

    # TOTAL_CHUNKS = 140
    # CURRENT_CHUNK = 0

    TRAINING_LIST = os.path.join(VGG2_PATH, "train_list.txt")
    # TEST_LIST = os.path.join(VGG2_PATH, "test_list.txt")

    # MAP ALL INDEXES

    indexes = map_indexes(TRAINING_LIST, TOTAL_CHUNKS)

    generate_tfrecord(
        os.path.join(VGG2_PATH, "train"),
        LANDMARK_PATH,
        TRAINING_LIST,
        os.path.join(
            args["<output-path>"], f"train_vgg2_chunk{CURRENT_CHUNK}.tfrecords"
        ),
        indexes[CURRENT_CHUNK],
    )

