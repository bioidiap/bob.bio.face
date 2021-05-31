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
    Base Transformer using the OpenCV interface.


    .. note::
       This class supports Caffe ``.caffemodel``, Tensorflow ``.pb``, Torch ``.t7`` ``.net``, Darknet ``.weights``, DLDT ``.bin``, and ONNX ``.onnx``


    Parameters
    ----------

    checkpoint_path: str
       Path containing the checkpoint

    config:
        Path containing some configuration file (e.g. .json, .prototxt)
    """

    def __init__(self, checkpoint_path=None, config=None, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.model = None

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

        import ipdb

        ipdb.set_trace()

        img = np.array(X)
        img = img / 255

        self.model.setInput(img)

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

        super(VGG16_Oxford, self).__init__(checkpoint_path, config)

    def _load_model(self):
        import cv2

        net = cv2.dnn.readNet(self.checkpoint_path, self.config)
        self.model = net
