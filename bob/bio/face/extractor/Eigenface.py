#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import numpy

import bob.learn.linear
import bob.io.base

from bob.bio.base.extractor import Extractor

import logging
logger = logging.getLogger("bob.bio.face")

class Eigenface (Extractor):
  """Extracts grid graphs from the images"""

  def __init__(self, subspace_dimension):
    # We have to register that this function will need a training step
    Extractor.__init__(self, requires_training = True, subspace_dimension = subspace_dimension)
    self.subspace_dimension = subspace_dimension


  def _check_data(self, data):
    assert isinstance(data, numpy.ndarray)
    assert data.ndim == 2
    assert data.dtype == numpy.float64


  def train(self, image_list, extractor_file):
    """Trains the eigenface extractor using the given list of training images"""
    [self._check_data(image) for image in image_list]

    # Initializes an array for the data
    data = numpy.vstack([image.flatten() for image in image_list])

    logger.info("  -> Training LinearMachine using PCA (SVD)")
    t = bob.learn.linear.PCATrainer()
    self.machine, __eig_vals = t.train(data)
    # Machine: get shape, then resize
    self.machine.resize(self.machine.shape[0], self.subspace_dimension)
    self.machine.save(bob.io.base.HDF5File(extractor_file, "w"))


  def load(self, extractor_file):
    # read PCA projector
    self.machine = bob.learn.linear.Machine(bob.io.base.HDF5File(extractor_file))


  def __call__(self, image):
    """Projects the data using the stored covariance matrix"""
    self._check_data(image)
    # Projects the data
    return self.machine(image.flatten())
