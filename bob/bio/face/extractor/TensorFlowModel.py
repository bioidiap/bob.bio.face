#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Yu Linghu & Xinyi Zhang <yu.linghu@uzh.ch, xinyi.zhang@uzh.ch>

import tensorflow as tf
from bob.extension import rc
from bob.learn.tensorflow.utils.image import to_channels_last
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from functools import partial
import pkg_resources
import os
import numpy as np
from tensorflow import keras

tf_model_directory = rc["bob.extractor_model.tf"]

class TensorFlowModel(TransformerMixin, BaseEstimator):
    """Extracts features using deep face recognition models under TensorFlow Interface.

  Users can download the pretrained face recognition models with TensorFlow Interface. The path to downloaded models should be specified before running the extractor (usually before running the pipeline file that includes the extractor). That is, set config of the model to :py:class:`bob.extractor_model.tf`. 
   
  .. code-block:: sh
  
    $ bob config set bob.extractor_model.tf /PATH/TO/MODEL/
  
  The extracted features can be combined with different the algorithms. 
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

        internal_path = pkg_resources.resource_filename(
            __name__, os.path.join("data", "resnet"),
        )

        checkpoint_path = (
            internal_path
            if rc["bob.bio.face.models.tfmodel"] is None
            else rc["bob.bio.face.models.tfmodel"]
        )

        self.checkpoint_path = checkpoint_path

    def _load_model(self):

        model = tf.keras.models.load_model(tf_model_directory)

        self.model = model

    def transform(self, X):
        """__call__(image) -> feature

    Extracts the features from the given image.

    **Parameters:**

    image : 2D :py:class:`numpy.ndarray` (floats)
      The image to extract the features from.

    **Returns:**

    feature : 2D or 3D :py:class:`numpy.ndarray` (floats)
      The list of features extracted from the image.
    """
    
        if self.model is None:
            self._load_model()
        
        X = check_array(X, allow_nd=True)
        X = tf.convert_to_tensor(X)
        X = to_channels_last(X)

        X = X/255
        predict = self.model.predict(X)

        return predict


    def __getstate__(self):
        # Handling unpicklable objects

        d = self.__dict__.copy()
        d["model"] = None
        return d

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}