#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Yu Linghu & Xinyi Zhang <yu.linghu@uzh.ch, xinyi.zhang@uzh.ch>

import bob.bio.base
from bob.bio.face.preprocessor import FaceCrop

from bob.bio.base.transformers.preprocessor import PreprocessorTransformer

import cv2
import numpy as np

from bob.learn.tensorflow.utils.image import to_channels_last
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array

from bob.extension import rc
from functools import partial
import pkg_resources
import os

from PIL import Image

opencv_model_directory = rc["bob.extractor_model.opencv"]
opencv_model_prototxt = rc["bob.extractor_weights.opencv"]


class OpenCVModel(TransformerMixin, BaseEstimator):
    """Extracts features using deep face recognition models under OpenCV Interface

  Users can download the pretrained face recognition models with OpenCV Interface. The path to downloaded models should be specified before running the extractor (usually before running the pipeline file that includes the extractor). That is, set config of the model frame to :py:class:`bob.extractor_model.opencv`, and set config of the parameters to :py:class:`bob.extractor_weights.opencv`. 
  
  .. code-block:: sh
  
    $ bob config set bob.extractor_model.opencv /PATH/TO/MODEL/
    $ bob config set bob.extractor_weights.opencv /PATH/TO/WEIGHTS/
  
  The extracted features can be combined with different the algorithms. 

    .. note::
       This structure only can be used for CAFFE pretrained model.
    """


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

        internal_path = pkg_resources.resource_filename(
            __name__, os.path.join("data", "opencv_model"),
        )

        checkpoint_path = (
            internal_path
            if rc["bob.bio.face.models.opencv"] is None
            else rc["bob.bio.face.models.opencv"]
        )

        self.checkpoint_path = checkpoint_path

    def _load_model(self):

        net = cv2.dnn.readNetFromCaffe(opencv_model_prototxt,opencv_model_directory)

        self.model = net

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

        img = np.array(X)
        img = img/255

        self.model.setInput(img)
     
        return self.model.forward()


    def __getstate__(self):
        # Handling unpicklable objects

        d = self.__dict__.copy()
        d["model"] = None
        return d

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}
