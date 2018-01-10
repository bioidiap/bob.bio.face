#!/usr/bin/env python
# encoding: utf-8

"""Features for face recognition"""

import numpy
import bob.io.base
from bob.bio.base.extractor import Extractor

import bob.io.base
import torch 
import torchvision.transforms as transforms
from torch.autograd import Variable

from bob.learn.pytorch.architectures import DRGAN_encoder as drgan_encoder

class LuanExtractor(Extractor):
  """

  **Parameters:**

  """

  def __init__(self):

    Extractor.__init__(self, skip_extractor_training=True)
    
    self.latent_dim = 320
    self.image_size = (3, 64, 64)
    self.encoder = drgan_encoder(input_image.shape, latent_dim)
  
    # image pre-processing
    self.to_tensor = transforms.ToTensor()
    self.norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


  def __call__(self, image):
    """__call__(image) -> feature

    Extract features

    **Parameters:**

    image : 3D :py:class:`numpy.ndarray` (floats)
      The image to extract the features from.

    **Returns:**

    feature : 2D :py:class:`numpy.ndarray` (floats)
      The extracted features
    """
    input_image = numpy.rollaxis(numpy.rollaxis(image, 2),2)
    input_image = self.to_tensor(input_image)
    input_image = self.norm(input_image)
    input_image = input_image.unsqueeze(0)
    encoded_id = encoder.forward(Variable(input_image))
    print encoded_id
    import sys
    sys.exit()
    return encoded_id

  # re-define the train function to get it non-documented
  def train(*args, **kwargs): raise NotImplementedError("This function is not implemented and should not be called.")

  def load(self, extractor_file):
    encoder.load_state_dict(torch.load(extractor_file, map_location=lambda storage, loc: storage)) 
