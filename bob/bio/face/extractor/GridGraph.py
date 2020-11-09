#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob.ip.gabor
import bob.io.base

import numpy
import math
from bob.bio.base.extractor import Extractor


class GridGraph(Extractor):
    """Extracts Gabor jets in a grid structure [GHW12]_ using functionalities from :ref:`bob.ip.gabor <bob.ip.gabor>`.

  The grid can be either aligned to the eye locations (in which case the grid might be rotated), or a fixed grid graph can be extracted.

  In the first case, the eye locations in the aligned image need to be provided.
  Additionally, the number of node between, along, above and below the eyes need to be specified.

  In the second case, a regular grid graph is created, by specifying the distance between two nodes.
  Additionally, the coordinate of the first node can be provided, which otherwise is calculated to evenly fill the whole image with nodes.

  **Parameters:**

  gabor_directions, gabor_scales, gabor_sigma, gabor_maximum_frequency, gabor_frequency_step, gabor_power_of_k, gabor_dc_free
    The parameters of the Gabor wavelet family, with its default values set as given in [WFK97]_.
    Please refer to :py:class:`bob.ip.gabor.Transform` for the documentation of these values.

  normalize_gabor_jets : bool
    Perform Gabor jet normalization during extraction?

  eyes : dict or ``None``
    If specified, the grid setup will be aligned to the eye positions {'reye' : (re_y, re_x), 'leye' : (le_y, le_x)}.
    Otherwise a regular grid graph will be extracted.

  nodes_between_eyes, nodes_along_eyes, nodes_above_eyes, nodes_below_eyes : int
    Only used when ``eyes`` is not ``None``.
    The number of nodes to be placed between, along, above or below the eyes.
    The final number of nodes will be: `(above + below + 1)` \\times (between + 2*along + 2).

  node_distance : (int, int)
    Only used when ``eyes`` is ``None``.
    The distance between two nodes in the regular grid graph.

  first_node : (int, int) or ``None``
    Only used when ``eyes`` is ``None``.
    If ``None``, it is calculated automatically to equally cover the whole image.
  """

    def __init__(
        self,
        # Gabor parameters
        gabor_directions=8,
        gabor_scales=5,
        gabor_sigma=2.0 * math.pi,
        gabor_maximum_frequency=math.pi / 2.0,
        gabor_frequency_step=math.sqrt(0.5),
        gabor_power_of_k=0,
        gabor_dc_free=True,
        # what kind of information to extract
        normalize_gabor_jets=True,
        # setup of the aligned grid
        eyes=None,  # if set, the grid setup will be aligned to the eye positions {'leye' : LEFT_EYE_POS, 'reye' : RIGHT_EYE_POS},
        nodes_between_eyes=4,
        nodes_along_eyes=2,
        nodes_above_eyes=3,
        nodes_below_eyes=7,
        # setup of static grid
        node_distance=None,  # one or two integral values
        first_node=None,  # one or two integral values, or None -> automatically determined
    ):

        # call base class constructor
        Extractor.__init__(
            self,
            gabor_directions=gabor_directions,
            gabor_scales=gabor_scales,
            gabor_sigma=gabor_sigma,
            gabor_maximum_frequency=gabor_maximum_frequency,
            gabor_frequency_step=gabor_frequency_step,
            gabor_power_of_k=gabor_power_of_k,
            gabor_dc_free=gabor_dc_free,
            normalize_gabor_jets=normalize_gabor_jets,
            eyes=eyes,
            nodes_between_eyes=nodes_between_eyes,
            nodes_along_eyes=nodes_along_eyes,
            nodes_above_eyes=nodes_above_eyes,
            nodes_below_eyes=nodes_below_eyes,
            node_distance=node_distance,
            first_node=first_node,
        )

        self.gabor_directions = gabor_directions
        self.gabor_scales = gabor_scales
        self.gabor_sigma = gabor_sigma
        self.gabor_maximum_frequency = gabor_maximum_frequency
        self.gabor_frequency_step = gabor_frequency_step
        self.gabor_power_of_k = gabor_power_of_k
        self.gabor_dc_free = gabor_dc_free
        self.normalize_gabor_jets = normalize_gabor_jets
        self.eyes = eyes
        self.nodes_between_eyes = nodes_between_eyes
        self.nodes_along_eyes = nodes_along_eyes
        self.nodes_above_eyes = nodes_above_eyes
        self.nodes_below_eyes = nodes_below_eyes
        self.node_distance = node_distance
        self.first_node = first_node

        self.normalize_jets = normalize_gabor_jets
        self.trafo_image = None
        self._init_non_pickables()

    def _init_non_pickables(self):

        # create Gabor wavelet transform class
        self.gwt = bob.ip.gabor.Transform(
            number_of_scales=self.gabor_scales,
            number_of_directions=self.gabor_directions,
            sigma=self.gabor_sigma,
            k_max=self.gabor_maximum_frequency,
            k_fac=self.gabor_frequency_step,
            power_of_k=self.gabor_power_of_k,
            dc_free=self.gabor_dc_free,
        )

        # create graph extractor
        if self.eyes is not None:
            self._aligned_graph = bob.ip.gabor.Graph(
                righteye=[int(e) for e in self.eyes["reye"]],
                lefteye=[int(e) for e in self.eyes["leye"]],
                between=int(self.nodes_between_eyes),
                along=int(self.nodes_along_eyes),
                above=int(self.nodes_above_eyes),
                below=int(self.nodes_below_eyes),
            )
        else:
            if self.node_distance is None:
                raise ValueError(
                    "Please specify either 'eyes' or the grid parameters 'node_distance' (and 'first_node')!"
                )
            self._aligned_graph = None
            self._last_image_resolution = None
            if isinstance(self.node_distance, (int, float)):
                self.node_distance = (int(self.node_distance), int(self.node_distance))

    def _extractor(self, image):
        """Creates an extractor based on the given image.
    If an aligned graph was specified in the constructor, it is simply returned.
    Otherwise the resolution of the given image is used to create a graph extractor.
    If the ``first_node`` was not specified, it is calculated automatically.
    """

        if self.trafo_image is None or self.trafo_image.shape[1:3] != image.shape:
            # create trafo image
            self.trafo_image = numpy.ndarray(
                (self.gwt.number_of_wavelets, image.shape[0], image.shape[1]),
                numpy.complex128,
            )

        if self._aligned_graph is not None:
            return self._aligned_graph

        # check if a new extractor needs to be created
        if self._last_image_resolution != image.shape:
            self._last_image_resolution = image.shape
            if self.first_node is None:
                # automatically compute the first node
                first_node = [0, 0]
                for i in (0, 1):
                    offset = int(
                        (
                            image.shape[i]
                            - int(image.shape[i] / self.node_distance[i])
                            * self.node_distance[i]
                        )
                        / 2
                    )
                    if (
                        offset < self.node_distance[i] // 2
                    ):  # This is not tested, but should ALWAYS be the case.
                        offset += self.node_distance[i] // 2
                    first_node[i] = offset
            else:
                first_node = self.first_node
            # .. and the last node
            last_node = tuple(
                [int(image.shape[i] - max(first_node[i], 1)) for i in (0, 1)]
            )

            # take the specified nodes
            self._graph = bob.ip.gabor.Graph(
                first=first_node, last=last_node, step=self.node_distance
            )

        return self._graph

    def __call__(self, image):
        """__call__(image) -> feature

    Returns a list of Gabor jets extracted from the given image.

    **Parameters:**

    image : 2D :py:class:`numpy.ndarray` (floats)
      The image to extract the features from.

    **Returns:**

    feature : [:py:class:`bob.ip.gabor.Jet`]
      The list of Gabor jets extracted from the image.
      The 2D location of the jet's nodes is not returned.
    """
        assert image.ndim == 2
        assert isinstance(image, numpy.ndarray)
        image = image.astype(numpy.float64)
        assert image.dtype == numpy.float64

        extractor = self._extractor(image)

        # perform Gabor wavelet transform
        self.gwt.transform(image, self.trafo_image)
        # extract face graph
        jets = extractor.extract(self.trafo_image)

        # normalize the Gabor jets of the graph only
        if self.normalize_jets:
            [j.normalize() for j in jets]

        # return the extracted face graph
        return self.__class__.serialize_jets(jets)

    def write_feature(self, feature, feature_file):
        """Writes the feature extracted by the `__call__` function to the given file.

    **Parameters:**

    feature : [:py:class:`bob.ip.gabor.Jet`]
      The list of Gabor jets extracted from the image.

    feature_file : str or :py:class:`bob.io.base.HDF5File`
      The name of the file or the file opened for writing.
    """
        feature_file = (
            feature_file
            if isinstance(feature_file, bob.io.base.HDF5File)
            else bob.io.base.HDF5File(feature_file, "w")
        )
        bob.ip.gabor.save_jets(feature, feature_file)

    def read_feature(self, feature_file):
        """read_feature(feature_file) -> feature

    Reads the feature written by the :py:meth:`write_feature` function from the given file.

    **Parameters:**

    feature_file : str or :py:class:`bob.io.base.HDF5File`
      The name of the file or the file opened for reading.

    **Returns:**

    feature : [:py:class:`bob.ip.gabor.Jet`]
      The list of Gabor jets read from file.
    """
        return self.__class__.serialize_jets(
            bob.ip.gabor.load_jets(bob.io.base.HDF5File(feature_file))
        )

    # re-define the train function to get it non-documented
    def train(*args, **kwargs):
        raise NotImplementedError(
            "This function is not implemented and should not be called."
        )

    def load(*args, **kwargs):
        pass

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("gwt")
        d.pop("_aligned_graph")
        if "_graph" in d:
            d.pop("_graph")
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._init_non_pickables()

    @staticmethod
    def serialize_jets(jets):
        serialize_jets = []
        for jet in jets:
            sj = bob.ip.gabor.Jet(jet.length)
            sj.jet = jet.jet
            serialize_jets.append(sj)
        return serialize_jets
