#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import tensorflow as tf
import os
from tensorflow.python import debug as tf_debug
import pkg_resources
import bob.extension.download
from bob.extension import rc
from sklearn.base import TransformerMixin, BaseEstimator

import logging

logger = logging.getLogger(__name__)


class TensorflowCompatV1(TransformerMixin, BaseEstimator):
    """
    Tensorflow v1 compatible set of transformers.

    Parameters
    ----------
    checkpoint_filename: str
        Path of your checkpoint. If the .meta file is providede the last checkpoint will be loaded.

    input_shape: tuple
        input_shape: Input shape for the tensorflow neural network

    architecture_fn :
        A tf.Tensor containing the operations to be executed
    """

    def __init__(self, checkpoint_filename, input_shape, architecture_fn):
        """Loads the tensorflow model
        """

        self.checkpoint_filename = checkpoint_filename
        self.input_shape = input_shape
        self.architecture_fn = architecture_fn
        self.loaded = False

    def transform(self, data):
        """
        Forward the data with the loaded neural network

        Parameters
        ----------
        image : numpy.ndarray
            Input Data

        Returns
        -------
        numpy.ndarray
            The features.

        """
        if not self.loaded:
            self.load_model()

        return self.session.run(
            self.embedding,
            feed_dict={self.input_tensor: data.reshape(self.input_shape)},
        )

    def load_model(self):
        logger.info(f"Loading model `{self.checkpoint_filename}`")

        tf.compat.v1.reset_default_graph()

        self.input_tensor = tf.placeholder(tf.float32, shape=self.input_shape)

        # Taking the embedding
        prelogits = self.architecture_fn(
            tf.stack(
                [
                    tf.image.per_image_standardization(i)
                    for i in tf.unstack(self.input_tensor)
                ]
            ),
            mode=tf.estimator.ModeKeys.PREDICT,
        )[0]
        self.embedding = tf.nn.l2_normalize(prelogits, dim=1, name="embedding")

        # Initializing the variables of the current architecture_fn
        self.session = tf.compat.v1.Session()
        self.session.run(tf.compat.v1.global_variables_initializer())

        # Loading the last checkpoint and overwriting the current variables
        saver = tf.compat.v1.train.Saver()
        if os.path.splitext(self.checkpoint_filename)[1] == ".meta":
            saver.restore(
                self.session,
                tf.train.latest_checkpoint(os.path.dirname(self.checkpoint_filename)),
            )
        elif os.path.isdir(self.checkpoint_filename):
            saver.restore(
                self.session, tf.train.latest_checkpoint(self.checkpoint_filename)
            )
        else:
            saver.restore(self.session, self.checkpoint_filename)

        self.loaded = True

    def __setstate__(self, d):
        # Handling unpicklable objects
        self.__dict__ = d
        self.loaded = False

    def __getstate__(self):
        # Handling unpicklable objects
        d = self.__dict__
        d.pop("session", None)
        d.pop("input_tensor", None)
        d.pop("embedding", None)
        tf.compat.v1.reset_default_graph()
        return d

    # def __del__(self):
    #    tf.compat.v1.reset_default_graph()

    def get_modelpath(self, bob_rc_variable, model_subdirectory):
        """
        Get default model path.

        First we try the to search this path via Global Configuration System.
        If we can not find it, we set the path in the directory
        `<project>/data`
        """

        # Priority to the RC path
        model_path = rc[bob_rc_variable]

        if model_path is None:

            model_path = pkg_resources.resource_filename(
                __name__, os.path.join("data", model_subdirectory)
            )

        return model_path

    def download_model(self, model_path, urls, zip_file="model.tar.gz"):
        """
        Download and unzip a model from some URL.

        Parameters
        ----------

        model_path: str
            Path where the model is supposed to be stored

        urls: list
            List of paths where the model is stored

        zip_file: str
            File name after the download

        """

        if not os.path.exists(model_path):
            bob.io.base.create_directories_safe(model_path)
            zip_file = os.path.join(model_path, zip_file)
            bob.extension.download.download_and_unzip(urls, zip_file)


    def fit(self, X, y=None):
        return self

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}

