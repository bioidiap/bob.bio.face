#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Thu May 24 10:41:42 CEST 2012
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import bob.io.base
import bob.ip.gabor

import unittest
import os
import numpy
import math
import tempfile
from nose.plugins.skip import SkipTest

import pkg_resources

regenerate_refs = False
seed_value = 5489


def test_gabor_jet():
  jets = bob.bio.base.load_resource("gabor-jet", "algorithm", preferred_package='bob.bio.face')
  assert isinstance(jets, bob.bio.face.algorithm.GaborJet)
  assert isinstance(jets, bob.bio.base.algorithm.Algorithm)
  assert not jets.performs_projection
  assert not jets.requires_projector_training
  assert not jets.use_projected_features_for_enrollment
  assert not jets.split_training_features_by_client
  assert not jets.requires_enroller_training

  # read input
  feature = bob.ip.gabor.load_jets(bob.io.base.HDF5File(pkg_resources.resource_filename("bob.bio.face.test", "data/graph_regular.hdf5")))

  # enroll
  model = jets.enroll([feature, feature])
  assert len(model) == len(feature)
  assert all(len(m) == 2 for m in model)
  assert all(model[n][i] == feature[n] for n in range(len(feature)) for i in range(2))

  # score
  assert abs(jets.score(model, feature) - 1.) < 1e-8
  assert abs(jets.score_for_multiple_probes(model, [feature, feature]) - 1.) < 1e-8


  # test averaging
  jets = bob.bio.face.algorithm.GaborJet(
    "PhaseDiffPlusCanberra",
    multiple_feature_scoring = "average_model"
  )
  model = jets.enroll([feature, feature])
  assert len(model) == len(feature)
  assert all(len(m) == 1 for m in model)

  # absoulte values must be identical
  assert all(numpy.allclose(model[n][0].abs, feature[n].abs) for n in range(len(model)))
  # phases might differ with 2 Pi
  for n in range(len(model)):
    for j in range(len(model[n][0].phase)):
      assert any(abs(model[n][0].phase[j] - feature[n].phase[j] - k*2.*math.pi) < 1e-5 for k in (0, -2, 2))

  assert abs(jets.score(model, feature) - 1.) < 1e-8
  assert abs(jets.score_for_multiple_probes(model, [feature, feature]) - 1.) < 1e-8


def test_histogram():
  histogram = bob.bio.base.load_resource("histogram", "algorithm", preferred_package='bob.bio.face')
  assert isinstance(histogram, bob.bio.face.algorithm.Histogram)
  assert isinstance(histogram, bob.bio.base.algorithm.Algorithm)
  assert not histogram.performs_projection
  assert not histogram.requires_projector_training
  assert not histogram.use_projected_features_for_enrollment
  assert not histogram.split_training_features_by_client
  assert not histogram.requires_enroller_training

  # read input
  feature1 = bob.bio.base.load(pkg_resources.resource_filename('bob.bio.face.test', 'data/lgbphs_sparse.hdf5'))
  feature2 = bob.bio.base.load(pkg_resources.resource_filename('bob.bio.face.test', 'data/lgbphs_with_phase.hdf5'))

  # enroll model from sparse features
  model1 = histogram.enroll([feature1, feature1])
  assert model1.shape == feature1.shape
  assert numpy.allclose(model1, feature1)

  # enroll from non-sparse features
  model2 = histogram.enroll([feature2, feature2])
  assert model2.shape == feature2.shape
  assert numpy.allclose(model2, feature2)

  # score without phase and sparse
  reference = 40960.
  assert abs(histogram.score(model1, feature1) - reference) < 1e-5
  assert abs(histogram.score_for_multiple_probes(model1, [feature1, feature1]) - reference) < 1e-5

  # score with phase, but non-sparse
  # reference doubles since we have two times more features
  reference *= 2.
  assert abs(histogram.score(model2, feature2) - reference) < 1e-5
  assert abs(histogram.score_for_multiple_probes(model2, [feature2, feature2]) - reference) < 1e-5


def test_bic_jets():
  bic = bob.bio.base.load_resource("bic-jets", "algorithm", preferred_package='bob.bio.face')
  assert isinstance(bic, bob.bio.base.algorithm.BIC)
  assert isinstance(bic, bob.bio.base.algorithm.Algorithm)

  # TODO: add more tests for bic-jets
