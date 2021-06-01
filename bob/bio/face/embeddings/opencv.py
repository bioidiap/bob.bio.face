#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Yu Linghu & Xinyi Zhang <yu.linghu@uzh.ch, xinyi.zhang@uzh.ch>
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import bob.bio.base
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
import os
from bob.extension.download import get_file


class OpenCVTransformer(TransformerMixin, BaseEstimator):
    """
    Base Transformer using the OpenCV DNN interface (https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html).


    .. note::
       This class supports Caffe ``.caffemodel``, Tensorflow ``.pb``, Torch ``.t7`` ``.net``, Darknet ``.weights``, DLDT ``.bin``, and ONNX ``.onnx``


    Parameters
    ----------

    checkpoint_path: str
       Path containing the checkpoint

    config:
        Path containing some configuration file (e.g. .json, .prototxt)

    preprocessor:
        A function that will transform the data right before forward. The default transformation is `X/255`

    """

    def __init__(
        self,
        checkpoint_path=None,
        config=None,
        preprocessor=lambda x: x / 255,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.model = None
        self.preprocessor = preprocessor

    def _load_model(self):
        import cv2

        net = cv2.dnn.readNet(self.checkpoint_path, self.config)
        self.model = net

    def transform(self, X):
        """__call__(image) -> feature

        Extracts the features from the given image.

        **Parameters:**

        X : 2D :py:class:`numpy.ndarray` (floats)
          The image to extract the features from.

        **Returns:**

        feature : 2D or 3D :py:class:`numpy.ndarray` (floats)
          The list of features extracted from the image.
        """

        import cv2

        if self.model is None:
            self._load_model()

        X = check_array(X, allow_nd=True)

        X = self.preprocessor(X)

        self.model.setInput(X)

        return self.model.forward()

    def __getstate__(self):
        # Handling unpicklable objects

        d = self.__dict__.copy()
        d["model"] = None
        return d

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}


class VGG16_Oxford(OpenCVTransformer):
    """
    Original VGG16 model from the paper: https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf

    """

    def __init__(self):
        urls = [
            "https://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz",
            "http://bobconda.lab.idiap.ch/public-upload/data/bob/bob.bio.face/master/caffe/vgg_face_caffe.tar.gz",
        ]

        filename = get_file(
            "vgg_face_caffe.tar.gz",
            urls,
            cache_subdir="data/caffe/vgg_face_caffe",
            file_hash="ee707ac6e890bc148cb155adeaad12be",
            extract=True,
        )
        path = os.path.dirname(filename)
        config = os.path.join(path, "vgg_face_caffe", "VGG_FACE_deploy.prototxt")
        checkpoint_path = os.path.join(path, "vgg_face_caffe", "VGG_FACE.caffemodel")

        caffe_average_img = [129.1863, 104.7624, 93.5940]

        def preprocessor(X):
            """
            Normalize using data from caffe

            Caffe has the shape `C x H x W` and the chanel is BGR and 

            """
            # To BGR
            X = X[:, ::-1, :, :].astype("float32")

            # Subtracting
            X[:, :, :, 0] -= caffe_average_img[0]
            X[:, :, :, 1] -= caffe_average_img[1]
            X[:, :, :, 2] -= caffe_average_img[2]

            return X

        super(VGG16_Oxford, self).__init__(checkpoint_path, config, preprocessor)

    def _load_model(self):
        import cv2

        net = cv2.dnn.readNet(self.checkpoint_path, self.config)
        self.model = net

