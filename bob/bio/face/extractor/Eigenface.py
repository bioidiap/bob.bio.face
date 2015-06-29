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
  """Performs a principal component analysis (PCA) on the given data.

  This algorithm computes a PCA projection (:py:class:`bob.learn.linear.PCATrainer`) on the given training images, and projects the images into face space.
  In opposition to :py:class:`bob.bio.base.algorithm.PCA`, here the eigenfces are used as features, i.e., to apply advanced face recognition algorithms on top of them.

  **Parameters:**

  subspace_dimension : int or float
    If specified as ``int``, defines the number of eigenvectors used in the PCA projection matrix.
    If specified as ``float`` (between 0 and 1), the number of eigenvectors is calculated such that the given percentage of variance is kept.

  kwargs : ``key=value`` pairs
    A list of keyword arguments directly passed to the :py:class:`bob.bio.base.extractor.Extractor` base class constructor.
  """

  def __init__(self, subspace_dimension):
    # We have to register that this function will need a training step
    Extractor.__init__(self, requires_training = True, subspace_dimension = subspace_dimension)
    self.subspace_dimension = subspace_dimension


  def _check_data(self, data):
    """Checks that the given data are appropriate."""
    assert isinstance(data, numpy.ndarray)
    assert data.ndim == 2
    assert data.dtype == numpy.float64


  def train(self, training_images, extractor_file):
    """Generates the PCA covariance matrix and writes it into the given extractor_file.

    Beforehand, all images are turned into a 1D pixel vector.

    **Parameters:**

    training_images : [2D :py:class:`numpy.ndarray`]
      A list of 2D training images to train the PCA projection matrix with.

    extractor_file : str
      A writable file, into which the PCA projection matrix (as a :py:class:`bob.learn.linear.Machine`) will be written.
    """
    [self._check_data(image) for image in training_images]

    # Initializes an array for the data
    data = numpy.vstack([image.flatten() for image in training_images])

    logger.info("  -> Training LinearMachine using PCA (SVD)")
    t = bob.learn.linear.PCATrainer()
    self.machine, variances = t.train(data)

    # compute variance percentage, if desired
    if isinstance(self.subspace_dimension, float):
      cummulated = numpy.cumsum(variances) / numpy.sum(variances)
      for index in range(len(cummulated)):
        if cummulated[index] > self.subspace_dimension:
          self.subspace_dimension = index
          break
      self.subspace_dimension = index
      logger.info("  -> Keeping %d eigenvectors" % self.subspace_dimension)

    # Machine: get shape, then resize
    self.machine.resize(self.machine.shape[0], self.subspace_dimension)
    self.machine.save(bob.io.base.HDF5File(extractor_file, "w"))


  def load(self, extractor_file):
    """Reads the PCA projection matrix from file.

    **Parameters:**

    extractor_file : str
      An existing file, from which the PCA projection matrix are read.
    """
    # read PCA projector
    self.machine = bob.learn.linear.Machine(bob.io.base.HDF5File(extractor_file))


  def __call__(self, image):
    """__call__(image) -> feature

    Projects the given image using the stored covariance matrix.

    Beforehand, the image is turned into a 1D pixel vector.

    **Parameters:**

    image : 2D :py:class:`numpy.ndarray` (floats)
      The image to extract the eigenface feature from.

    **Returns:**

    feature : 1D :py:class:`numpy.ndarray` (floats)
      The extracted eigenface feature.
    """
    self._check_data(image)
    # Projects the data
    return self.machine(image.flatten())
