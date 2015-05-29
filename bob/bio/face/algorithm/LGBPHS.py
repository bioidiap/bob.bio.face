#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob.math

import numpy

from bob.bio.base.algorithm import Algorithm

class LGBPHS (Algorithm):
  """Tool chain for computing local Gabor binary pattern histogram sequences"""

  def __init__(
      self,
      distance_function = bob.math.chi_square,
      is_distance_function = True,
      multiple_probe_scoring = 'average'
  ):
    """Initializes the local Gabor binary pattern histogram sequence tool"""

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
    """Enrolling model by taking the average of all features"""
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


  def score(self, model, probe):
    """Computes the score using the specified histogram measure; returns a similarity value (bigger -> better)"""
    sparse = self._is_sparse(probe)
    self._check_feature(model, sparse)
    self._check_feature(probe, sparse)

    if sparse:
      # assure that the probe is sparse as well
      return self.factor * self.distance_function(model[0,:], model[1,:], probe[0,:], probe[1,:])
    else:
      return self.factor * self.distance_function(model, probe)
