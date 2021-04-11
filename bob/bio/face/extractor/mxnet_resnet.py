#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Yu Linghu & Xinyi Zhang <yu.linghu@uzh.ch, xinyi.zhang@uzh.ch>

"""Feature extraction resnet models using mxnet interface"""
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
import numpy as np
import pkg_resources
import os
import mxnet as mx
from mxnet import gluon
import warnings
from bob.extension import rc
mxnet_resnet_directory = rc["bob.extractor_model.mxnet"]
mxnet_weight_directory = rc["bob.extractor_weights.mxnet"]

class mxnet_model(TransformerMixin, BaseEstimator):

    """Extracts features using deep face recognition models under MxNet Interfaces.
  
  Users can download the pretrained face recognition models with MxNet Interface. The path to downloaded models should be specified before running the extractor (usually before running the pipeline file that includes the extractor). That is, set config of the model frame to :py:class:`bob.extractor_model.mxnet`, and set config of the parameters to :py:class:`bob.extractor_weights.mxnet`.
  
  .. code-block:: sh
  
    $ bob config set bob.extractor_model.mxnet /PATH/TO/MODEL/
    $ bob config set bob.extractor_weights.mxnet /PATH/TO/WEIGHTS/
  
  Examples: (Pretrained ResNet models): `LResNet100E-IR,ArcFace@ms1m-refine-v2 <https://github.com/deepinsight/insightface>`_  
  
  The extracted features can be combined with different the algorithms.  

  **Parameters:**
  use_gpu: True or False.
    """

    def __init__(self, use_gpu=False, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.use_gpu = use_gpu

        internal_path = pkg_resources.resource_filename(
            __name__, os.path.join("data", "resnet"),
        )

        checkpoint_path = (
            internal_path
            if rc["bob.bio.face.models.mxnet_resnet"] is None
            else rc["bob.bio.face.models.mxnet_resnet"]
        )

        self.checkpoint_path = checkpoint_path

    def _load_model(self):

        ctx = mx.gpu() if self.use_gpu else mx.cpu()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            deserialized_net = gluon.nn.SymbolBlock.imports(mxnet_resnet_directory, ['data'], mxnet_weight_directory, ctx=ctx)
               
        self.model = deserialized_net

    def transform(self, X):
        """__call__(image) -> feature

    Extracts the features from the given image.

    **Parameters:**

    image : 2D :py:class:`numpy.ndarray` (floats)
      The image to extract the features from.

    **Returns:**

    feature : 2D, 3D, or 4D :py:class:`numpy.ndarray` (floats)
      The list of features extracted from the image.
    """
    
        if self.model is None:
            self._load_model()

        X = check_array(X, allow_nd=True)
        X = mx.nd.array(X)

        return self.model(X,).asnumpy()


    def __getstate__(self):
        # Handling unpicklable objects

        d = self.__dict__.copy()
        d["model"] = None
        return d

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}