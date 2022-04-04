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

import numpy
import math

import pkg_resources

regenerate_refs = False
seed_value = 5489


def test_histogram():
    histogram = bob.bio.face.algorithm.Histogram(
        distance_function=bob.math.histogram_intersection, is_distance_function=False
    )

    assert isinstance(histogram, bob.bio.face.algorithm.Histogram)
    assert isinstance(histogram, bob.bio.base.algorithm.Algorithm)
    assert not histogram.performs_projection
    assert not histogram.requires_projector_training
    assert not histogram.use_projected_features_for_enrollment
    assert not histogram.split_training_features_by_client
    assert not histogram.requires_enroller_training

    # read input
    feature1 = bob.bio.base.load(
        pkg_resources.resource_filename("bob.bio.face.test", "data/lgbphs_sparse.hdf5")
    )
    feature2 = bob.bio.base.load(
        pkg_resources.resource_filename(
            "bob.bio.face.test", "data/lgbphs_with_phase.hdf5"
        )
    )

    # enroll model from sparse features
    model1 = histogram.enroll([feature1, feature1])
    assert model1.shape == feature1.shape
    assert numpy.allclose(model1, feature1)

    # enroll from non-sparse features
    model2 = histogram.enroll([feature2, feature2])
    assert model2.shape == feature2.shape
    assert numpy.allclose(model2, feature2)

    # score without phase and sparse
    reference = 40960.0
    assert abs(histogram.score(model1, feature1) - reference) < 1e-5
    assert (
        abs(
            histogram.score_for_multiple_probes(model1, [feature1, feature1])
            - reference
        )
        < 1e-5
    )

    # score with phase, but non-sparse
    # reference doubles since we have two times more features
    reference *= 2.0
    assert abs(histogram.score(model2, feature2) - reference) < 1e-5
    assert (
        abs(
            histogram.score_for_multiple_probes(model2, [feature2, feature2])
            - reference
        )
        < 1e-5
    )
