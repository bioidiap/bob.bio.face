#!/usr/bin/env python
# coding: utf-8

"""
Converts the MSCeleb annotated version to TF RECORD

Usage:
    msceleb_to_tfrecord.py <msceleb-path> <output-path> [--keys=<arg> --image-size=<arg> --use-eyes]
    msceleb_to_tfrecord.py -h | --help

Options:
  -h --help             Show this screen.
  --keys=<arg>          Pickle with the keys
  --image-size=<arg>    Final image size [default: 126]
  --use-eyes            Use eyes annotations. If not set, it will use the face crop only

"""


from docopt import docopt
import numpy as np
import os
import bob.io.image
import bob.io.base
import tensorflow as tf
import sys
from datetime import datetime
import pickle
import numpy
from bob.bio.face.preprocessor import FaceCrop


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def detect_mtcnn_margin_face_crop(annotations, image, margin=44, final_size=126):
    """
    Face crop using bounding box
    """

    annotations["topleft"] = [annotations["topleft"][0], annotations["topleft"][1]]
    annotations["bottomright"] = [
        annotations["bottomright"][0],
        annotations["bottomright"][1],
    ]

    annotations["topleft"][0] = numpy.uint(
        numpy.maximum(annotations["topleft"][0] - margin / 2, 0)
    )
    annotations["topleft"][1] = numpy.uint(
        numpy.maximum(annotations["topleft"][1] - margin / 2, 0)
    )

    annotations["bottomright"][0] = numpy.uint(
        numpy.minimum(annotations["bottomright"][0] + margin / 2, image.shape[1])
    )
    annotations["bottomright"][1] = numpy.uint(
        numpy.minimum(annotations["bottomright"][1] + margin / 2, image.shape[2])
    )

    cropped_positions = {"topleft": (0, 0), "bottomright": (final_size, final_size)}
    cropper = FaceCrop(
        cropped_image_size=(final_size, final_size),
        cropped_positions=cropped_positions,
        color_channel="rgb",
        fixed_positions=None,
        annotator=None,
    )

    detected = cropper.transform([image], [annotations])[0]

    return detected.astype("uint8")


def detect_mtcnn_margin_eyes(annotations, image, margin=44, final_size=126):

    # final image position w.r.t the image size
    RIGHT_EYE_POS = (final_size / 3.44, final_size / 3.02)
    LEFT_EYE_POS = (final_size / 3.44, final_size / 1.49)

    # RIGHT_EYE_POS = (final_size / 3.34,
    #                 final_size / 3.02)
    # LEFT_EYE_POS = (final_size / 3.44,
    #                final_size / 1.59)

    cropped_positions = {"leye": LEFT_EYE_POS, "reye": RIGHT_EYE_POS}

    cropper = FaceCrop(
        cropped_image_size=(final_size, final_size),
        cropped_positions=cropped_positions,
        color_channel="rgb",
        fixed_positions=None,
        annotator=None,
    )

    detected = cropper.transform([image], [annotations])[0]

    return detected.astype("uint8")


def generate_tfrecord(
    chunk_path,
    output_tf_record_path,
    detector,
    keys,
    final_size=126,
    margin=44,
    use_eyes=False,
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
                if os.path.splitext(image_path)[-1] != ".png":
                    continue
                image = bob.io.image.load(image_path)
                annotations = detector.annotations(image)

                if len(annotations) == 0:
                    continue
                else:
                    # Getting the first annotation
                    annotations = annotations[0]

                if use_eyes:
                    detected_image = detect_mtcnn_margin_eyes(
                        annotations, image, margin=margin, final_size=final_size
                    )
                else:

                    detected_image = detect_mtcnn_margin_face_crop(
                        annotations, image, margin=margin, final_size=final_size
                    )
                # Converting H x W x C
                detected_image = bob.io.image.to_matplotlib(detected_image)

                write_single_line_tfrecord(
                    tf_writer, detected_image, keys[identity], identity
                )


def get_keys(base_path, all_chunks):
    """
    Read the file structure from `annotations.csv` to get the samples properly annotated
    """

    def decide(annotations_path):
        """
        Decide if we should consider an identity or not.
        The annotation has the following format.

        ```
        0,3.png,4.png,1
        1,4.png,40.png,1
        2,40.png,46.png,1
        3,46.png,47.png,1
        4,47.png,55.png,1
        5,55.png,56.png,1
        6,56.png,71.png,1
        7,71.png,79.png,1
        8,79.png,99.png,1
        9,99.png,100.png,1
        10,100.png,3.png,1
        ```

        The last collumn can assume the values:
          - `0`: The pair of images are NOT from the same identity
          - `1`: The pair of images ARE from the same identity
          - `2`: The annotator doesn't know what to say


        Here I'm considering the identity if you have more than 75% `1` and  `2`.

        """

        with open(annotations_path) as f:
            lines = 0
            lines_with_zero = 0
            for l in f.readlines():
                lines += 1
                if l.split(",")[-1] == "\n":
                    lines_with_zero += 1
                    continue

                decision = int(l.split(",")[-1])
                if decision == 0:
                    lines_with_zero += 1

        # Discarting identities with more than 50% of the pairs not
        # considered from the same identity
        # This is the first cut
        return True if lines_with_zero / lines < 0.50 else False

    offset = 0
    keys = dict()
    for chunk in all_chunks:
        path = os.path.join(base_path, chunk)
        for identity in os.listdir(path):
            if not os.path.isdir(os.path.join(path, identity)):
                continue

            statistics = os.path.join(path, identity, "annotations.csv")

            if decide(statistics):
                keys[identity] = offset
                offset += 1
            else:
                print(f"Rejected {identity}")
    return keys


if __name__ == "__main__":
    args = docopt(__doc__)

    MSCELEB_PATH = args["<msceleb-path>"]
    output_path = args["<output-path>"]
    image_size = int(args["--image-size"])
    use_eyes = args["--use-eyes"]
    margin = 0

    if "SGE_TASK_LAST" in os.environ:
        TOTAL_CHUNKS = int(os.environ["SGE_TASK_LAST"])
        CURRENT_CHUNK = int(os.environ["SGE_TASK_ID"]) - 1
    else:
        TOTAL_CHUNKS = 1
        CURRENT_CHUNK = 0

    all_chunks = [f"chunk_{i}" for i in range(43)]

    if args["--keys"] is None:
        keys = get_keys(MSCELEB_PATH, all_chunks)
        with open("keys.pickle", "wb") as f:
            f.write(pickle.dumps(keys))
    else:
        keys = pickle.loads(open(args["--keys"], "rb").read())

    chunk_id = all_chunks[CURRENT_CHUNK]

    from bob.bio.face.annotator import MTCNN

    detector = MTCNN()

    output_tf_record_path = os.path.join(output_path, chunk_id + ".tfrecords")

    generate_tfrecord(
        os.path.join(MSCELEB_PATH, chunk_id),
        output_tf_record_path,
        detector,
        keys,
        final_size=image_size,
        margin=margin,
        use_eyes=use_eyes,
    )

    sys.stdout.write("Done \n")
    sys.stdout.flush()

