#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Yu Linghu & Xinyi Zhang <yu.linghu@uzh.ch, xinyi.zhang@uzh.ch>

import torch
from bob.learn.tensorflow.utils.image import to_channels_last
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
from bob.extension import rc
from functools import partial
import pkg_resources
import os
import numpy as np
import imp

pytorch_model_directory = rc["bob.extractor_model.pytorch"]
pytorch_weight_directory = rc["bob.extractor_weights.pytorch"]

class pytorch_loaded_model(TransformerMixin, BaseEstimator):
    """Extracts features using deep face recognition models under PyTorch Interface, especially for the models and weights that need to load by hand.
    
  Users can download the pretrained face recognition models with PyTorch Interface. The path to downloaded models should be specified before running the extractor (usually before running the pipeline file that includes the extractor). That is, set config of the model frame to :py:class:`bob.extractor_model.pytorch`, and set config of the parameters to :py:class:`bob.extractor_weights.pytorch`. 
  
  .. code-block:: sh
  
    $ bob config set bob.extractor_model.pytorch /PATH/TO/MODEL/
    $ bob config set bob.extractor_weights.pytorch /PATH/TO/WEIGHTS/
  
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
            if rc["bob.bio.face.models.pytorchmodel"] is None
            else rc["bob.bio.face.models.pytorchmodel"]
        )

        self.checkpoint_path = checkpoint_path
        self.device = None

    def _load_model(self):

        MainModel = imp.load_source('MainModel', pytorch_model_directory)
        network = torch.load(pytorch_weight_directory)
        network.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        network.to(self.device)

        self.model = network

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

        X = torch.Tensor(X)

        return self.model(X).detach().numpy()


    def __getstate__(self):
        # Handling unpicklable objects

        d = self.__dict__.copy()
        d["model"] = None
        return d

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}
        
        
        
        
        
        
        
        
class pytorch_library_model(TransformerMixin, BaseEstimator):
    """Extracts features using deep face recognition with registered model frames in the PyTorch Library. 
    
  Users can import the pretrained face recognition models from PyTorch library. The model should be called in the pipeline. Example: `facenet_pytorch <https://github.com/timesler/facenet-pytorch>`_

  The extracted features can be combined with different the algorithms.  

  **Parameters:**
  model: pytorch model calling from library.
  use_gpu: True or False.
    """

    def __init__(self, model=None, use_gpu=False, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.use_gpu = use_gpu

        internal_path = pkg_resources.resource_filename(
            __name__, os.path.join("data", "resnet"),
        )

        checkpoint_path = (
            internal_path
            if rc["bob.bio.face.models.pytorchmodel"] is None
            else rc["bob.bio.face.models.pytorchmodel"]
        )

        self.checkpoint_path = checkpoint_path
        self.device = None

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

        X = torch.Tensor(X)

        return self.model(X).detach().numpy()


    def __getstate__(self):
        # Handling unpicklable objects

        d = self.__dict__.copy()
        d["model"] = None
        return d

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}