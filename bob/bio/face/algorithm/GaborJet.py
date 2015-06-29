#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob.ip.gabor
import bob.io.base

import numpy
import math

from bob.bio.base.algorithm import Algorithm

class GaborJet (Algorithm):
  """Computes a comparison of lists of Gabor jets using a similarity function of :py:class:`bob.ip.gabor.Similarity`.

  The model enrollment simply stores all extracted Gabor jets for all enrollment features.
  By default (i.e., ``multiple_feature_scoring = 'max_jet'``), the scoring uses an advanced local strategy.
  For each node, the similarity between the given probe jet and all model jets is computed, and only the *highest* value is kept.
  These values are finally averaged over all node positions.
  Other strategies can be obtained using a different ``multiple_feature_scoring``.

  **Parameters:**

  gabor_jet_similarity_type : str:
    The type of Gabor jet similarity to compute.
    Please refer to the documentation of :py:class:`bob.ip.gabor.Similarity` for a list of possible values.

  multiple_feature_scoring : str
    How to fuse the local similarities into a single similarity value.
    Possible values are:

    * ``'average_model'`` : During enrollment, an average model is computed using functionality of :ref:`bob.ip.gabor <bob.ip.gabor>`.
    * ``'average'`` : For each node, the average similarity is computed. Finally, the average of those similarities is returned.
    * ``'min_jet', 'max_jet', 'med_jet'`` : For each node, the minimum, maximum or median similarity is computed. Finally, the average of those similarities is returned.
    * ``'min_graph', 'max_graph', 'med_graph'`` : For each node, the average similarity is computed. Finally, the minimum, maximum or median of those similarities is returned.

  gabor_directions, gabor_scales, gabor_sigma, gabor_maximum_frequency, gabor_frequency_step, gabor_power_of_k, gabor_dc_free
    These parameters are required by the disparity-based Gabor jet similarity functions, see :py:class:`bob.ip.gabor.Similarity`..
    The default values are identical to the ones in the :py:class:`bob.bio.face.extractor.GridGraph`.
    Please assure that this class and the :py:class:`bob.bio.face.extractor.GridGraph` class get the same configuration, otherwise unexpected things might happen.
  """

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
    """enroll(enroll_features) -> model

    Enrolls the model using one of several strategies.
    Commonly, the bunch graph strategy [WFK97]_ is applied, by storing several Gabor jets for each node.

    When ``multiple_feature_scoring = 'average_model'``, for each node the average :py:class:`bob.ip.gabor.Jet` is computed.
    Otherwise, all enrollment jets are stored, grouped by node.

    **Parameters:**

    enroll_features : [[:py:class:`bob.ip.gabor.Jet`]]
      The list of enrollment features.
      Each sub-list contains a full graph.

    **Returns:**

    model : [[:py:class:`bob.ip.gabor.Jet`]]
      The enrolled model.
      Each sub-list contains a list of jets, which correspond to the same node.
      When ``multiple_feature_scoring = 'average_model'`` each sub-list contains a single :py:class:`bob.ip.gabor.Jet`.
    """
    [self._check_feature(feature) for feature in enroll_features]
    assert len(enroll_features)
    assert all(len(feature) == len(enroll_features[0]) for feature in enroll_features)

    # re-organize the jets to have a collection of jets per node
    jets_per_node = [[enroll_features[g][n] for g in range(len(enroll_features))] for n in range(len(enroll_features[0]))]

    if self.jet_scoring is not None:
      return jets_per_node

    # compute average model, and keep a list with a single jet per node
    return [[bob.ip.gabor.Jet(jets_per_node[n])] for n in range(len(jets_per_node))]


  def write_model(self, model, model_file):
    """Writes the model enrolled by the :py:meth:`enroll` function to the given file.

    **Parameters:**

    model : [[:py:class:`bob.ip.gabor.Jet`]]
      The enrolled model.

    model_file : str or :py:class:`bob.io.base.HDF5File`
      The name of the file or the file opened for writing.
    """
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
    """read_model(model_file) -> model

    Reads the model written by the :py:meth:`write_model` function from the given file.

    **Parameters:**

    model_file : str or :py:class:`bob.io.base.HDF5File`
      The name of the file or the file opened for reading.

    **Returns:**

    model : [[:py:class:`bob.ip.gabor.Jet`]]
      The list of Gabor jets read from file.
    """
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
    """read_probe(probe_file) -> probe

    Reads the probe file, e.g., as written by the :py:meth:`bob.bio.face.extractor.GridGraph.write_feature` function from the given file.

    **Parameters:**

    probe_file : str or :py:class:`bob.io.base.HDF5File`
      The name of the file or the file opened for reading.

    **Returns:**

    probe : [:py:class:`bob.ip.gabor.Jet`]
      The list of Gabor jets read from file.
    """
    return bob.ip.gabor.load_jets(bob.io.base.HDF5File(probe_file))


  def score(self, model, probe):
    """score(model, probe) -> score

    Computes the score of the probe and the model using the desired Gabor jet similarity function and the desired score fusion strategy.

    **Parameters:**

    model : [[:py:class:`bob.ip.gabor.Jet`]]
      The model enrolled by the :py:meth:`enroll` function.

    probe : [:py:class:`bob.ip.gabor.Jet`]
      The probe read by the :py:meth:`read_probe` function.

    **Returns:**

    score : float
      The fused similarity score.
    """
    self._check_feature(probe)
    [self._check_feature(m) for m in model]
    assert len(model) == len(probe)

    # select jet score averaging function
    jet_scoring = numpy.average if self.jet_scoring is None else self.jet_scoring
    graph_scoring = numpy.average if self.graph_scoring is None else self.graph_scoring
    local_scores = [jet_scoring([self.similarity_function(m, pro) for m in mod]) for mod, pro in zip(model, probe)]
    return graph_scoring(local_scores)


  def score_for_multiple_probes(self, model, probes):
    """score(model, probes) -> score

    This function computes the score between the given model graph(s) and several given probe graphs.
    The same local scoring strategy as for several model jets is applied, but this time the local scoring strategy is applied between all graphs from the model and probes.

    **Parameters:**

    model : [[:py:class:`bob.ip.gabor.Jet`]]
      The model enrolled by the :py:meth:`enroll` function.
      The sub-lists are groups by nodes.

    probes : [[:py:class:`bob.ip.gabor.Jet`]]
      A list of probe graphs.
      The sub-lists are groups by graph.

    **Returns:**

    score : float
      The fused similarity score.
    """
    [self._check_feature(probe) for probe in probes]
    [self._check_feature(m) for m in model]
    assert all(len(model) == len(probe) for probe in probes)

    jet_scoring = numpy.average if self.jet_scoring is None else self.jet_scoring
    graph_scoring = numpy.average if self.graph_scoring is None else self.graph_scoring
    local_scores = [jet_scoring([self.similarity_function(m, probe[n]) for m in model[n] for probe in probes]) for n in range(len(model))]
    return graph_scoring(local_scores)


  # overwrite functions to avoid them being documented.
  def train_projector(*args, **kwargs) : raise NotImplementedError("This function is not implemented and should not be called.")
  def load_projector(*args, **kwargs) : pass
  def project(*args, **kwargs) : raise NotImplementedError("This function is not implemented and should not be called.")
  def write_feature(*args, **kwargs) : raise NotImplementedError("This function is not implemented and should not be called.")
  def read_feature(*args, **kwargs) : raise NotImplementedError("This function is not implemented and should not be called.")
  def train_enroller(*args, **kwargs) : raise NotImplementedError("This function is not implemented and should not be called.")
  def load_enroller(*args, **kwargs) : pass
  def score_for_multiple_models(*args, **kwargs) : raise NotImplementedError("This function is not implemented and should not be called.")
