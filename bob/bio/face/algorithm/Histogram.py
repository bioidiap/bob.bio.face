#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob.math

import numpy

from bob.bio.base.algorithm import Algorithm

class Histogram (Algorithm):
  """Computes the distance between histogram sequences.

  Both sparse and non-sparse representations of histograms are supported.
  For enrollment, to date only the averaging of histograms is implemented.

  **Parameters:**

  distance_function : function
    The function to be used to compare two histograms.
    This function should accept sparse histograms.

  is_distance_function : bool
    Is the given ``distance_function`` distance function (lower values are better) or a similarity function (higher values are better)?

  multiple_probe_scoring : str or ``None``
    The way, scores are fused when multiple probes are available.
    See :py:func:`bob.bio.base.score_fusion_strategy` for possible values.
  """

  def __init__(
      self,
      distance_function = bob.math.chi_square,
      is_distance_function = True,
      multiple_probe_scoring = 'average'
  ):

    # call base class constructor
    Algorithm.__init__(
        self,

        distance_function = str(distance_function),
        is_distance_function = is_distance_function,

        multiple_model_scoring = None,
        multiple_probe_scoring = multiple_probe_scoring
    )

    # remember distance function
    self.distance_function = distance_function
    self.factor =  -1. if is_distance_function else 1


  def _is_sparse(self, feature):
    assert isinstance(feature, numpy.ndarray)
    return feature.ndim == 2

  def _check_feature(self, feature, sparse):
    assert isinstance(feature, numpy.ndarray)
    if sparse:
      # check that we have a 2D array
      assert feature.ndim == 2
      assert feature.shape[0] == 2
    else:
      assert feature.ndim == 1


  def enroll(self, enroll_features):
    """enroll(enroll_features) -> model

    Enrolls a model by taking the average of all histograms.

    enroll_features : [1D or 2D :py:class:`numpy.ndarray`]
      The histograms that should be averaged.
      Histograms can be specified sparse (2D) or non-sparse (1D)

    **Returns:**

    model : 1D or 2D :py:class:`numpy.ndarray`
      The averaged histogram, sparse  (2D) or non-sparse (1D).
    """
    assert len(enroll_features)
    sparse = self._is_sparse(enroll_features[0])
    [self._check_feature(feature, sparse) for feature in enroll_features]

    if sparse:
      # get all indices for the sparse model
      values = {}
      # iterate through all sparse features
      for feature in enroll_features:
        # collect the values by index
        for j in range(feature.shape[1]):
          index = int(feature[0,j])
          value = feature[1,j] / float(len(enroll_features))
          # add up values
          if index in values:
            values[index] += value
          else:
            values[index] = value

      # create model containing all the used indices
      model = numpy.ndarray((2, len(values)), dtype = numpy.float64)
      for i, index in enumerate(sorted(values.keys())):
        model[0,i] = index
        model[1,i] = values[index]
    else:
      model = numpy.zeros(enroll_features[0].shape, dtype = numpy.float64)
      # add up models
      for feature in enroll_features:
        model += feature
      # normalize by number of models
      model /= float(len(enroll_features))

    # return averaged model
    return model


  def read_probe(self, probe_file):
    """read_probe(probe_file) -> probe

    Reads the probe feature from the given file.

    **Parameters:**

    probe_file : str or :py:class:`bob.io.base.HDF5File`
      The file (open for reading) or the name of an existing file to read from.

    **Returns:**

    probe : 1D or 2D :py:class:`numpy.ndarray`
      The probe read by the :py:meth:`read_probe` function.
    """
    return bob.bio.base.load(probe_file)


  def score(self, model, probe):
    """score(model, probe) -> score

    Computes the score of the probe and the model using the desired histogram distance function.
    The resulting score is the negative distance, if ``is_distance_function = True``.
    Both sparse and non-sparse models and probes are accepted, but their sparseness must agree.

    **Parameters:**

    model : 1D or 2D :py:class:`numpy.ndarray`
      The model enrolled by the :py:meth:`enroll` function.

    probe : 1D or 2D :py:class:`numpy.ndarray`
      The probe read by the :py:meth:`read_probe` function.

    **Returns:**

    score : float
      The resulting similarity score.
    """
    sparse = self._is_sparse(probe)
    self._check_feature(model, sparse)
    self._check_feature(probe, sparse)

    if sparse:
      # assure that the probe is sparse as well
      return self.factor * self.distance_function(model[0,:], model[1,:], probe[0,:], probe[1,:])
    else:
      return self.factor * self.distance_function(model, probe)


  # overwrite functions to avoid them being documented.
  def train_projector(*args, **kwargs) : raise NotImplementedError("This function is not implemented and should not be called.")
  def load_projector(*args, **kwargs) : pass
  def project(*args, **kwargs) : raise NotImplementedError("This function is not implemented and should not be called.")
  def write_feature(*args, **kwargs) : raise NotImplementedError("This function is not implemented and should not be called.")
  def read_feature(*args, **kwargs) : raise NotImplementedError("This function is not implemented and should not be called.")
  def train_enroller(*args, **kwargs) : raise NotImplementedError("This function is not implemented and should not be called.")
  def load_enroller(*args, **kwargs) : pass
  def score_for_multiple_models(*args, **kwargs) : raise NotImplementedError("This function is not implemented and should not be called.")
