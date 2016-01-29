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

import bob.bio.base
import bob.bio.face

import unittest
import os
import numpy
import math
from nose.plugins.skip import SkipTest

import bob.io.base.test_utils
from bob.bio.base.test import utils

import pkg_resources

regenerate_refs = False

def _compare(data, reference, write_function = bob.bio.base.save, read_function = bob.bio.base.load, atol = 1e-5, rtol = 1e-8):
  # write reference?
  if regenerate_refs:
    write_function(data, reference)

  # compare reference
  reference = read_function(reference)
  assert numpy.allclose(data, reference, atol=atol, rtol=rtol)

def _data():
  return bob.bio.base.load(pkg_resources.resource_filename('bob.bio.face.test', 'data/cropped.hdf5'))


def test_dct_blocks():
  # read input
  data = _data()
  dct = bob.bio.base.load_resource('dct-blocks', 'extractor', preferred_package='bob.bio.face')
  assert isinstance(dct, bob.bio.face.extractor.DCTBlocks)
  assert isinstance(dct, bob.bio.base.extractor.Extractor)
  assert not dct.requires_training

  # generate smaller extractor, using mixed tuple and int input for the block size and overlap
  dct = bob.bio.face.extractor.DCTBlocks(8, (0,0), 15)

  # extract features
  feature = dct(data)
  assert feature.ndim == 2
  # feature dimension is one lower than the block size, since blocks are normalized by default
  assert feature.shape == (80, 14)
  reference = pkg_resources.resource_filename('bob.bio.face.test', 'data/dct_blocks.hdf5')
  _compare(feature, reference, dct.write_feature, dct.read_feature)


def test_graphs():
  data = _data()
  graph = bob.bio.base.load_resource('grid-graph', 'extractor', preferred_package='bob.bio.face')
  assert isinstance(graph, bob.bio.face.extractor.GridGraph)
  assert isinstance(graph, bob.bio.base.extractor.Extractor)
  assert not graph.requires_training

  # generate smaller extractor, using mixed tuple and int input for the node distance and first location
  graph = bob.bio.face.extractor.GridGraph(node_distance = 24)

  # extract features
  feature = graph(data)

  reference = pkg_resources.resource_filename('bob.bio.face.test', 'data/graph_regular.hdf5')
  # write reference?
  if regenerate_refs:
    graph.write_feature(feature, reference)

  # compare reference
  reference = graph.read_feature(reference)
  assert len(reference) == len(feature)
  assert all(isinstance(f, bob.ip.gabor.Jet) for f in feature)
  assert all(numpy.allclose(r.jet, f.jet) for r,f in zip(reference, feature))


  # get reference face graph extractor
  cropper = bob.bio.base.load_resource('face-crop-eyes', 'preprocessor', preferred_package='bob.bio.face')
  eyes = cropper.cropped_positions
  # generate aligned graph extractor
  graph = bob.bio.face.extractor.GridGraph(
    # setup of the aligned grid
    eyes = eyes,
    nodes_between_eyes = 4,
    nodes_along_eyes = 2,
    nodes_above_eyes = 2,
    nodes_below_eyes = 7
  )

  nodes = graph._extractor(data).nodes
  assert len(nodes) == 100
  assert numpy.allclose(nodes[22], eyes['reye'])
  assert numpy.allclose(nodes[27], eyes['leye'])

  assert nodes[0] < eyes['reye']
  assert nodes[-1] > eyes['leye']


def test_lgbphs():
  data = _data()
  lgbphs = bob.bio.base.load_resource('lgbphs', 'extractor', preferred_package='bob.bio.face')
  assert isinstance(lgbphs, bob.bio.face.extractor.LGBPHS)
  assert isinstance(lgbphs, bob.bio.base.extractor.Extractor)
  assert not lgbphs.requires_training

  # in this test, we use a smaller setup of the LGBPHS features
  lgbphs = bob.bio.face.extractor.LGBPHS(
      block_size = 8,
      block_overlap = 0,
      gabor_directions = 4,
      gabor_scales = 2,
      gabor_sigma = math.sqrt(2.) * math.pi,
      sparse_histogram = True
  )

  # extract feature
  feature = lgbphs(data)
  assert feature.ndim == 2

  reference = pkg_resources.resource_filename('bob.bio.face.test', 'data/lgbphs_sparse.hdf5')
  _compare(feature, reference, lgbphs.write_feature, lgbphs.read_feature)

  # generate new non-sparse extractor including Gabor phases
  lgbphs = bob.bio.face.extractor.LGBPHS(
      block_size = 8,
      block_overlap = 0,
      gabor_directions = 4,
      gabor_scales = 2,
      gabor_sigma = math.sqrt(2.) * math.pi,
      use_gabor_phases = True
  )
  feature = lgbphs(data)
  assert feature.ndim == 1

  reference = pkg_resources.resource_filename('bob.bio.face.test', 'data/lgbphs_with_phase.hdf5')
  _compare(feature, reference, lgbphs.write_feature, lgbphs.read_feature)


def test_eigenface():
  temp_file = bob.io.base.test_utils.temporary_filename()
  data = _data()
  eigen1 = bob.bio.base.load_resource('eigenface', 'extractor', preferred_package='bob.bio.face')
  assert isinstance(eigen1, bob.bio.face.extractor.Eigenface)
  assert isinstance(eigen1, bob.bio.base.extractor.Extractor)
  assert eigen1.requires_training

  # create extractor with a smaller number of kept eigenfaces
  train_data = utils.random_training_set(data.shape, 400, 0., 255.)
  eigen2 = bob.bio.face.extractor.Eigenface(subspace_dimension = 5)
  reference = pkg_resources.resource_filename('bob.bio.face.test', 'data/eigenface_extractor.hdf5')
  try:
    # train the projector
    eigen2.train(train_data, temp_file)

    assert os.path.exists(temp_file)

    if regenerate_refs: shutil.copy(temp_file, reference_file)

    # check projection matrix
    eigen1.load(reference)
    eigen2.load(temp_file)

    assert eigen1.machine.shape == eigen2.machine.shape
    for i in range(5):
      assert numpy.abs(eigen1.machine.weights[:,i] - eigen2.machine.weights[:,i] < 1e-5).all() or numpy.abs(eigen1.machine.weights[:,i] + eigen2.machine.weights[:,i] < 1e-5).all()

  finally:
    if os.path.exists(temp_file): os.remove(temp_file)

  # now, we can execute the extractor and check that the feature is still identical
  feature = eigen1(data)
  assert feature.ndim == 1
  reference = pkg_resources.resource_filename('bob.bio.face.test', 'data/eigenface_feature.hdf5')
  _compare(feature, reference, eigen1.write_feature, eigen1.read_feature)


"""
  def test05_sift_key_points(self):
    # check if VLSIFT is available
    import bob.ip.base
    if not hasattr(bob.ip.base, "VLSIFT"):
      raise SkipTest("VLSIFT is not part of bob.ip.base; maybe SIFT headers aren't installed in your system?")

    # we need the preprocessor tool to actually read the data
    preprocessor = facereclib.preprocessing.Keypoints()
    data = preprocessor.read_data(self.input_dir('key_points.hdf5'))
    # now, we extract features from it
    extractor = self.config('sift')
    feature = self.execute(extractor, data, 'sift.hdf5', epsilon=1e-4)
    self.assertEqual(len(feature.shape), 1)


"""
