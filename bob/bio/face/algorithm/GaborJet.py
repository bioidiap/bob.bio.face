#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob.ip.gabor
import bob.io.base

import numpy
import math

from bob.bio.base.algorithm import Algorithm

class GaborJet (Algorithm):
  """Algorithm chain for computing Gabor jets, Gabor graphs, and Gabor graph comparisons"""

  def __init__(
      self,
      # parameters for the tool
      gabor_jet_similarity_type,
      multiple_feature_scoring = 'max_jet',
      # some similarity functions might need a GaborWaveletTransform class, so we have to provide the parameters here as well...
      gabor_directions = 8,
      gabor_scales = 5,
      gabor_sigma = 2. * math.pi,
      gabor_maximum_frequency = math.pi / 2.,
      gabor_frequency_step = math.sqrt(.5),
      gabor_power_of_k = 0,
      gabor_dc_free = True
  ):

    # call base class constructor
    Algorithm.__init__(
        self,

        gabor_jet_similarity_type = gabor_jet_similarity_type,
        multiple_feature_scoring = multiple_feature_scoring,
        gabor_directions = gabor_directions,
        gabor_scales = gabor_scales,
        gabor_sigma = gabor_sigma,
        gabor_maximum_frequency = gabor_maximum_frequency,
        gabor_frequency_step = gabor_frequency_step,
        gabor_power_of_k = gabor_power_of_k,
        gabor_dc_free = gabor_dc_free,

        multiple_model_scoring = None,
        multiple_probe_scoring = None
    )

    # the Gabor wavelet transform; used by (some of) the Gabor jet similarities
    gwt = bob.ip.gabor.Transform(
        number_of_scales = gabor_scales,
        number_of_directions = gabor_directions,
        sigma = gabor_sigma,
        k_max = gabor_maximum_frequency,
        k_fac = gabor_frequency_step,
        power_of_k = gabor_power_of_k,
        dc_free = gabor_dc_free
    )

    # jet comparison function
    self.similarity_function = bob.ip.gabor.Similarity(gabor_jet_similarity_type, gwt)

    # how to proceed with multiple features per model
    self.jet_scoring = {
        'average_model' : None, # compute an average model
        'average' : numpy.average, # compute the average similarity
        'min_jet' : min, # for each jet location, compute the minimum similarity
        'max_jet' : max, # for each jet location, compute the maximum similarity
        'med_jet' : numpy.median, # for each jet location, compute the median similarity
        'min_graph' : numpy.average, # for each model graph, compute the minimum average similarity
        'max_graph' : numpy.average, # for each model graph, compute the maximum average similarity
        'med_graph' : numpy.average, # for each model graph, compute the median average similarity
    }[multiple_feature_scoring]

    self.graph_scoring = {
        'average_model' : None, # compute an average model
        'average' : numpy.average, # compute the average similarity
        'min_jet' : numpy.average, # for each jet location, compute the minimum similarity
        'max_jet' : numpy.average, # for each jet location, compute the maximum similarity
        'med_jet' : numpy.average, # for each jet location, compute the median similarity
        'min_graph' : min, # for each model graph, compute the minimum average similarity
        'max_graph' : max, # for each model graph, compute the maximum average similarity
        'med_graph' : numpy.median, # for each model graph, compute the median average similarity
    }[multiple_feature_scoring]


  def _check_feature(self, feature):
    assert isinstance(feature, list)
    assert len(feature)
    assert all(isinstance(f, bob.ip.gabor.Jet) for f in feature)

  def enroll(self, enroll_features):
    """Enrolls the model by computing an average graph for each model"""
    [self._check_feature(feature) for feature in enroll_features]
    assert len(enroll_features)
    assert all(len(feature) == len(enroll_features[0]) for feature in enroll_features)

    # re-organize the jets to have a collection of jets per node
    jets_per_node = [[enroll_features[g][n] for g in range(len(enroll_features))] for n in range(len(enroll_features[0]))]

    if self.jet_scoring is not None:
      return jets_per_node

    # compute average model, and keep a list with a single jet per node
    return [[bob.ip.gabor.Jet(jets_per_node[n])] for n in range(len(jets_per_node))]


  def save_model(self, model, model_file):
    """Saves the enrolled model of Gabor jets to file."""
    f = bob.io.base.HDF5File(model_file, 'w')
    # several model graphs
    f.set("NumberOfNodes", len(model))
    for g in range(len(model)):
      name = "Node-" + str(g+1)
      f.create_group(name)
      f.cd(name)
      bob.ip.gabor.save_jets(model[g], f)
      f.cd("..")
    f.close()

  def read_model(self, model_file):
    f = bob.io.base.HDF5File(model_file)
    count = f.get("NumberOfNodes")
    model = []
    for g in range(count):
      name = "Node-" + str(g+1)
      f.cd(name)
      model.append(bob.ip.gabor.load_jets(f))
      f.cd("..")
    return model


  def read_probe(self, probe_file):
    return bob.ip.gabor.load_jets(bob.io.base.HDF5File(probe_file))


  def score(self, model, probe):
    """Computes the score of the probe and the model"""
    self._check_feature(probe)
    [self._check_feature(m) for m in model]
    assert len(model) == len(probe)

    # select jet score averaging function
    jet_scoring = numpy.average if self.jet_scoring is None else self.jet_scoring
    graph_scoring = numpy.average if self.graph_scoring is None else self.graph_scoring
    local_scores = [jet_scoring([self.similarity_function(m, pro) for m in mod]) for mod, pro in zip(model, probe)]
    return graph_scoring(local_scores)


  def score_for_multiple_probes(self, model, probes):
    """This function computes the score between the given model graph(s) and several given probe graphs."""
    [self._check_feature(probe) for probe in probes]
    graph_scoring = numpy.average if self.graph_scoring is None else self.graph_scoring
    return graph_scoring([self.score(model, probe) for probe in probes])
