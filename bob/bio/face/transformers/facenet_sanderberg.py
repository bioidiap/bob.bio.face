#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
Wrapper for the free FaceNet variant:
    https://github.com/davidsandberg/facenet


    Model 20170512-110547    
"""

from __future__ import division

from sklearn.base import TransformerMixin, BaseEstimator

import os
import re
import logging
import numpy as np
import tensorflow as tf
from bob.ip.color import gray_to_rgb
from bob.io.image import to_matplotlib
from bob.extension import rc
import bob.extension.download
import bob.io.base

logger = logging.getLogger(__name__)

FACENET_MODELPATH_KEY = "bob.bio.face.facenet_sanderberg_modelpath"


def prewhiten(img):
    mean = np.mean(img)
    std = np.std(img)
    std_adj = np.maximum(std, 1.0 / np.sqrt(img.size))
    y = np.multiply(np.subtract(img, mean), 1 / std_adj)
    return y


def get_model_filenames(model_dir):
    # code from https://github.com/davidsandberg/facenet
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith(".meta")]
    if len(meta_files) == 0:
        raise ValueError("No meta file found in the model directory (%s)" % model_dir)
    elif len(meta_files) > 1:
        raise ValueError(
            "There should not be more than one meta file in the model "
            "directory (%s)" % model_dir
        )
    meta_file = meta_files[0]
    max_step = -1
    for f in files:
        step_str = re.match(r"(^model-[\w\- ]+.ckpt-(\d+))", f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


class FaceNetSanderberg(TransformerMixin, BaseEstimator):
    """Wrapper for the free FaceNet variant:
    https://github.com/davidsandberg/facenet

    And for a preprocessor you can use::

        from bob.bio.face.preprocessor import FaceCrop
        # This is the size of the image that this model expects
        CROPPED_IMAGE_HEIGHT = 160
        CROPPED_IMAGE_WIDTH = 160
        # eye positions for frontal images
        RIGHT_EYE_POS = (46, 53)
        LEFT_EYE_POS = (46, 107)
        # Crops the face using eye annotations
        preprocessor = FaceCrop(
            cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
            cropped_positions={'leye': LEFT_EYE_POS, 'reye': RIGHT_EYE_POS},
            color_channel='rgb'
        )

    """

    def __init__(
        self,
        model_path=rc[FACENET_MODELPATH_KEY],
        image_size=160,
        layer_name="embeddings:0",
        **kwargs,
    ):
        super(FaceNetSanderberg, self).__init__()
        self.model_path = model_path
        self.image_size = image_size
        self.layer_name = layer_name
        self.loaded = False
        self._clean_unpicklables()

    def _clean_unpicklables(self):
        self.session = None
        self.embeddings = None
        self.graph = None
        self.images_placeholder = None
        self.phase_train_placeholder = None

    def _check_feature(self, img):
        img = np.asarray(img)

        def _convert(img):
            assert img.shape[-2] == self.image_size
            assert img.shape[-3] == self.image_size
            img = prewhiten(img)
            return img

        if img.ndim == 3:
            img = np.moveaxis(img, 0, -1)
            return _convert(img)[None, ...]  # Adding another axis
        elif img.ndim == 4:
            img = np.moveaxis(img, 1, -1)
            return _convert(img)
        else:
            raise ValueError(f"Image shape {img.shape} not supported")

    def load_model(self):
        tf.compat.v1.reset_default_graph()

        session_conf = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
        )
        self.graph = tf.Graph()
        self.session = tf.compat.v1.Session(graph=self.graph, config=session_conf)

        if self.model_path is None:
            self.model_path = self.get_modelpath()
        if not os.path.exists(self.model_path):
            bob.io.base.create_directories_safe(FaceNetSanderberg.get_modelpath())
            zip_file = os.path.join(
                FaceNetSanderberg.get_modelpath(), "20170512-110547.zip"
            )
            urls = [
                # This link only works in Idiap CI to save bandwidth.
                "http://www.idiap.ch/private/wheels/gitlab/"
                "facenet_model2_20170512-110547.zip",
                # this link to dropbox would work for everybody
                "https://www.dropbox.com/s/"
                "k7bhxe58q7d48g7/facenet_model2_20170512-110547.zip?dl=1",
            ]
            bob.extension.download.download_and_unzip(urls, zip_file)

        # code from https://github.com/davidsandberg/facenet
        model_exp = os.path.expanduser(self.model_path)

        with self.graph.as_default():
            if os.path.isfile(model_exp):
                logger.info("Model filename: %s" % model_exp)
                with tf.compat.v1.gfile.FastGFile(model_exp, "rb") as f:
                    graph_def = tf.compat.v1.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name="")
            else:
                logger.info("Model directory: %s" % model_exp)
                meta_file, ckpt_file = get_model_filenames(model_exp)

                logger.info("Metagraph file: %s" % meta_file)
                logger.info("Checkpoint file: %s" % ckpt_file)

                saver = tf.compat.v1.train.import_meta_graph(
                    os.path.join(model_exp, meta_file)
                )
                saver.restore(self.session, os.path.join(model_exp, ckpt_file))
        # Get input and output tensors
        self.images_placeholder = self.graph.get_tensor_by_name("input:0")
        self.embeddings = self.graph.get_tensor_by_name(self.layer_name)
        self.phase_train_placeholder = self.graph.get_tensor_by_name("phase_train:0")
        logger.info("Successfully loaded the model.")
        self.loaded = True

    def transform(self, X, **kwargs):
        def _transform(X):

            images = self._check_feature(X)
            if not self.loaded:
                self.load_model()

            feed_dict = {
                self.images_placeholder: images,
                self.phase_train_placeholder: False,
            }
            features = self.session.run(self.embeddings, feed_dict=feed_dict)
            return features

        if isinstance(X, list):
            return [_transform(i) for i in X]
        else:
            return _transform(X)

    @staticmethod
    def get_modelpath():
        """
        Get default model path.

        First we try the to search this path via Global Configuration System.
        If we can not find it, we set the path in the directory
        `<project>/data`
        """

        # Priority to the RC path
        model_path = rc[FACENET_MODELPATH_KEY]

        if model_path is None:
            import pkg_resources

            model_path = pkg_resources.resource_filename(
                __name__, "data/FaceNet/20170512-110547"
            )

        return model_path

    def __setstate__(self, d):
        # Handling unpicklable objects
        self.__dict__ = d
        self.loaded = False

    def __getstate__(self):
        # Handling unpicklable objects
        d = self.__dict__
        d.pop("session") if "session" in d else None
        d.pop("embeddings") if "embeddings" in d else None
        d.pop("graph") if "graph" in d else None
        d.pop("images_placeholder") if "images_placeholder" in d else None
        d.pop("phase_train_placeholder") if "phase_train_placeholder" in d else None
        tf.compat.v1.reset_default_graph()
        return d

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}

    def fit(self, X, y=None):
        return self
