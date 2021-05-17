#!/usr/bin/env python
# coding: utf-8

"""
Crop VGG2 with loose crop based on bounding box

Usage:
    vgg2_2_tfrecords.py <vgg-path> <output-path>  [--factor=<kn>]
    vgg2_2_tfrecords.py -h | --help

Options:
  -h --help             Show this screen.  
  --factor=<kn>         Crop Factor [default: 0.3]

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
                landmarks = np.array([float(line[i]) for i in [1, 2, 3, 4]])
                return {
                    "topleft": (landmarks[1], landmarks[0]),
                    "dimensions": (landmarks[3], landmarks[2]),
                    "bottomright": (
                        landmarks[1] + landmarks[3],
                        landmarks[0] + landmarks[2],
                    ),
                }

        else:
            return None


def extend_annotations(annotations, img_bottom_right, factor=0.3):
    width = annotations["dimensions"][1]
    height = annotations["dimensions"][0]

    new_annotations = {"topleft": [0, 0], "bottomright": [0, 0]}

    new_annotations["topleft"][0] = max(0, annotations["topleft"][0] - height * factor)
    new_annotations["topleft"][1] = max(0, annotations["topleft"][1] - width * factor)

    new_annotations["bottomright"][0] = min(
        img_bottom_right[1], annotations["bottomright"][0] + height * factor
    )
    new_annotations["bottomright"][1] = min(
        img_bottom_right[0], annotations["bottomright"][1] + width * factor
    )

    return new_annotations


from bob.bio.face.preprocessor import FaceCrop


def align(image, annotations, cropped_image_size=(126, 126), factor=0.3):

    cropped_image_height, cropped_image_width = cropped_image_size

    img_bottom_right = (image.shape[1], image.shape[2])
    new_annotations = extend_annotations(annotations, img_bottom_right, factor=factor)

    cropped_positions = {"topleft": (0, 0), "bottomright": cropped_image_size}
    cropper = FaceCrop(
        cropped_image_size=cropped_image_size,
        cropped_positions=cropped_positions,
        color_channel="rgb",
        fixed_positions=None,
        annotator=None,
    )
    return bob.io.image.to_matplotlib(
        cropper.transform([image], [new_annotations])[0]
    ).astype("uint8")


def get_id_by_line(line):
    return line.split("/")[0]


def generate_tfrecord(
    base_path, landmark_path, file_list, output_tf_record_path, indexes, factor=0.3
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
                    if landmarks is None:
                        raise ValueError(f"Landmark for {file_name} not found!")

                    # aligned_image = align(img, annotations)
                    aligned_image = align(
                        img, landmarks, factor=factor, cropped_image_size=(126, 126)
                    )
                    bob.io.base.save(bob.io.image.to_bob(aligned_image), "xuucu.png")
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
    LANDMARK_PATH = os.path.join(VGG2_PATH, "bb_landmark", "loose_bb_train.csv")

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
        factor=float(args["--factor"]),
    )

