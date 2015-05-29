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
import facereclib
from nose.plugins.skip import SkipTest

import pkg_resources

regenerate_refs = False
seed_value = 5489


def test_gabor_jet():
  jets = bob.bio.base.load_resource("gabor-jet", "algorithm")
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


def test_lgbphs():
  lgbphs = bob.bio.base.load_resource("lgbphs", "algorithm")
  assert isinstance(lgbphs, bob.bio.face.algorithm.LGBPHS)
  assert isinstance(lgbphs, bob.bio.base.algorithm.Algorithm)
  assert not lgbphs.performs_projection
  assert not lgbphs.requires_projector_training
  assert not lgbphs.use_projected_features_for_enrollment
  assert not lgbphs.split_training_features_by_client
  assert not lgbphs.requires_enroller_training

  # read input
  feature1 = bob.bio.base.load(pkg_resources.resource_filename('bob.bio.face.test', 'data/lgbphs_sparse.hdf5'))
  feature2 = bob.bio.base.load(pkg_resources.resource_filename('bob.bio.face.test', 'data/lgbphs_with_phase.hdf5'))

  # enroll model from sparse features
  model1 = lgbphs.enroll([feature1, feature1])
  assert model1.shape == feature1.shape
  assert numpy.allclose(model1, feature1)

  # enroll from non-sparse features
  model2 = lgbphs.enroll([feature2, feature2])
  assert model2.shape == feature2.shape
  assert numpy.allclose(model2, feature2)

  # score without phase and sparse
  reference = 40960.
  assert abs(lgbphs.score(model1, feature1) - reference) < 1e-5
  assert abs(lgbphs.score_for_multiple_probes(model1, [feature1, feature1]) - reference) < 1e-5

  # score with phase, but non-sparse
  # reference doubles since we have two times more features
  reference *= 2.
  assert abs(lgbphs.score(model2, feature2) - reference) < 1e-5
  assert abs(lgbphs.score_for_multiple_probes(model2, [feature2, feature2]) - reference) < 1e-5


"""
  def test09_plda(self):
    # read input
    feature = facereclib.utils.load(self.input_dir('linearize.hdf5'))
    # assure that the config file is readable
    tool = self.config('pca+plda')
    self.assertTrue(isinstance(tool, facereclib.tools.PLDA))

    # here, we use a reduced complexity for test purposes
    tool = facereclib.tools.PLDA(
        subspace_dimension_of_f = 2,
        subspace_dimension_of_g = 2,
        subspace_dimension_pca = 10,
        plda_training_iterations = 1,
        INIT_SEED = seed_value,
    )
    self.assertFalse(tool.performs_projection)
    self.assertTrue(tool.requires_enroller_training)

    # train the projector
    t = tempfile.mkstemp('pca+plda.hdf5', prefix='frltest_')[1]
    tool.train_enroller(facereclib.utils.tests.random_training_set_by_id(feature.shape, count=20, minimum=0., maximum=255.), t)
    if regenerate_refs:
      import shutil
      shutil.copy2(t, self.reference_dir('pca+plda_enroller.hdf5'))

    # load the projector file
    tool.load_enroller(self.reference_dir('pca+plda_enroller.hdf5'))
    # compare the resulting machines
    test_file = bob.io.base.HDF5File(t)
    test_file.cd('/pca')
    pca_machine = bob.learn.linear.Machine(test_file)
    test_file.cd('/plda')
    plda_machine = bob.learn.em.PLDABase(test_file)
    # TODO: compare the PCA machines
    #self.assertEqual(pca_machine, tool.m_pca_machine)
    # TODO: compare the PLDA machines
    #self.assertEqual(plda_machine, tool.m_plda_base_machine)
    os.remove(t)

    # enroll model
    model = tool.enroll([feature])
    if regenerate_refs:
      model.save(bob.io.base.HDF5File(self.reference_dir('pca+plda_model.hdf5'), 'w'))
    # TODO: compare the models with the reference
    #reference_model = tool.read_model(self.reference_dir('pca+plda_model.hdf5'))
    #self.assertEqual(model, reference_model)

    # score
    sim = tool.score(model, feature)
    self.assertAlmostEqual(sim, 0.)
    # score with a concatenation of the probe
    self.assertAlmostEqual(tool.score_for_multiple_probes(model, [feature, feature]), 0.)

"""
