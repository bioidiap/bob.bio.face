#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob.ip.gabor
import bob.ip.base

import numpy
import math

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
from bob.pipelines.sample import SampleBatch


class LGBPHS(TransformerMixin, BaseEstimator):
    """Extracts *Local Gabor Binary Pattern Histogram Sequences* (LGBPHS) [ZSG05]_ from the images, using functionality from :ref:`bob.ip.base <bob.ip.base>` and :ref:`bob.ip.gabor <bob.ip.gabor>`.

  The block size and the overlap of the blocks can be varied, as well as the parameters of the Gabor wavelet (:py:class:`bob.ip.gabor.Transform`) and the LBP extractor (:py:class:`bob.ip.base.LBP`).

  **Parameters:**

  block_size : int or (int, int)
    The size of the blocks that will be extracted.
    This parameter might be either a single integral value, or a pair ``(block_height, block_width)`` of integral values.

  block_overlap : int or (int, int)
    The overlap of the blocks in vertical and horizontal direction.
    This parameter might be either a single integral value, or a pair ``(block_overlap_y, block_overlap_x)`` of integral values.
    It needs to be smaller than the ``block_size``.

  gabor_directions, gabor_scales, gabor_sigma, gabor_maximum_frequency, gabor_frequency_step, gabor_power_of_k, gabor_dc_free
    The parameters of the Gabor wavelet family, with its default values set as given in [WFK97]_.
    Please refer to :py:class:`bob.ip.gabor.Transform` for the documentation of these values.

  use_gabor_phases : bool
    Extract also the Gabor phases (inline) and not only the absolute values.
    In this case, Extended LGBPHS features [ZSQ09]_ will be extracted.

  lbp_radius, lbp_neighbor_count, lbp_uniform, lbp_circular, lbp_rotation_invariant, lbp_compare_to_average, lbp_add_average
    The parameters of the LBP.
    Please see :py:class:`bob.ip.base.LBP` for the documentation of these values.

    .. note::
       The default values are as given in [ZSG05]_ (the values of [ZSQ09]_ might differ).

  sparse_histogram : bool
    If specified, the histograms will be handled in a sparse way.
    This reduces the size of the extracted features, but the computation will take longer.

    .. note::
       Sparse histograms are only supported, when ``split_histogram = None``.

  split_histogram : one of ``('blocks', 'wavelets', 'both')`` or ``None``
    Defines, how the histogram sequence is split.
    This could be interesting, if the histograms should be used in another way as simply concatenating them into a single histogram sequence (the default).
  """

    def __init__(
        self,
        # Block setup
        block_size,  # one or two parameters for block size
        block_overlap=0,  # one or two parameters for block overlap
        # Gabor parameters
        gabor_directions=8,
        gabor_scales=5,
        gabor_sigma=2.0 * math.pi,
        gabor_maximum_frequency=math.pi / 2.0,
        gabor_frequency_step=math.sqrt(0.5),
        gabor_power_of_k=0,
        gabor_dc_free=True,
        use_gabor_phases=False,
        # LBP parameters
        lbp_radius=2,
        lbp_neighbor_count=8,
        lbp_uniform=True,
        lbp_circular=True,
        lbp_rotation_invariant=False,
        lbp_compare_to_average=False,
        lbp_add_average=False,
        # histogram options
        sparse_histogram=False,
        split_histogram=None,
    ):
        # block parameters
        self.block_size = (
            block_size
            if isinstance(block_size, (tuple, list))
            else (block_size, block_size)
        )
        self.block_overlap = (
            block_overlap
            if isinstance(block_overlap, (tuple, list))
            else (block_overlap, block_overlap)
        )
        if (
            self.block_size[0] < self.block_overlap[0]
            or self.block_size[1] < self.block_overlap[1]
        ):
            raise ValueError(
                "The overlap is bigger than the block size. This won't work. Please check your setup!"
            )

        self.gabor_directions = gabor_directions
        self.gabor_scales = gabor_scales
        self.gabor_sigma = gabor_sigma
        self.gabor_maximum_frequency = gabor_maximum_frequency
        self.gabor_frequency_step = gabor_frequency_step
        self.gabor_power_of_k = gabor_power_of_k
        self.gabor_dc_free = gabor_dc_free
        self.use_gabor_phases = use_gabor_phases
        self.lbp_radius = lbp_radius
        self.lbp_neighbor_count = lbp_neighbor_count
        self.lbp_uniform = lbp_uniform
        self.lbp_circular = lbp_circular
        self.lbp_rotation_invariant = lbp_rotation_invariant
        self.lbp_compare_to_average = lbp_compare_to_average
        self.lbp_add_average = lbp_add_average
        self.sparse_histogram = sparse_histogram
        self.split_histogram = split_histogram

        self._init_non_pickables()

    def _init_non_pickables(self):
        # Gabor wavelet transform class
        self.gwt = bob.ip.gabor.Transform(
            number_of_scales=self.gabor_scales,
            number_of_directions=self.gabor_directions,
            sigma=self.gabor_sigma,
            k_max=self.gabor_maximum_frequency,
            k_fac=self.gabor_frequency_step,
            power_of_k=self.gabor_power_of_k,
            dc_free=self.gabor_dc_free,
        )
        self.trafo_image = None
        self.use_phases = self.use_gabor_phases

        self.lbp = bob.ip.base.LBP(
            neighbors=self.lbp_neighbor_count,
            radius=float(self.lbp_radius),
            circular=self.lbp_circular,
            to_average=self.lbp_compare_to_average,
            add_average_bit=self.lbp_add_average,
            uniform=self.lbp_uniform,
            rotation_invariant=self.lbp_rotation_invariant,
            border_handling="wrap",
        )

        self.split = self.split_histogram
        self.sparse = self.sparse_histogram
        if self.sparse and self.split:
            raise ValueError("Sparse histograms cannot be split! Check your setup!")

    def _fill(self, lgbphs_array, lgbphs_blocks, j):
        """Copies the given array into the given blocks"""
        # fill array in the desired shape
        if self.split is None:
            start = j * self.n_bins * self.n_blocks
            for b in range(self.n_blocks):
                lgbphs_array[
                    start + b * self.n_bins : start + (b + 1) * self.n_bins
                ] = lgbphs_blocks[b][:]
        elif self.split == "blocks":
            for b in range(self.n_blocks):
                lgbphs_array[
                    b, j * self.n_bins : (j + 1) * self.n_bins
                ] = lgbphs_blocks[b][:]
        elif self.split == "wavelets":
            for b in range(self.n_blocks):
                lgbphs_array[
                    j, b * self.n_bins : (b + 1) * self.n_bins
                ] = lgbphs_blocks[b][:]
        elif self.split == "both":
            for b in range(self.n_blocks):
                lgbphs_array[j * self.n_blocks + b, 0 : self.n_bins] = lgbphs_blocks[b][
                    :
                ]

    def _sparsify(self, array):
        """This function generates a sparse histogram from a non-sparse one."""
        if not self.sparse:
            return array
        if len(array.shape) == 2 and array.shape[0] == 2:
            # already sparse
            return array
        assert len(array.shape) == 1
        indices = []
        values = []
        for i in range(array.shape[0]):
            if array[i] != 0.0:
                indices.append(i)
                values.append(array[i])
        return numpy.array([indices, values], dtype=numpy.float64)

    def transform(self, X):
        """__call__(image) -> feature

    Extracts the local Gabor binary pattern histogram sequence from the given image.

    **Parameters:**

    image : 2D :py:class:`numpy.ndarray` (floats)
      The image to extract the features from.

    **Returns:**

    feature : 2D or 3D :py:class:`numpy.ndarray` (floats)
      The list of Gabor jets extracted from the image.
      The 2D location of the jet's nodes is not returned.
    """
    
        def _extract(image):
            assert image.ndim == 2
            assert isinstance(image, numpy.ndarray)
            assert image.dtype == numpy.float64

            # perform GWT on image
            if self.trafo_image is None or self.trafo_image.shape[1:3] != image.shape:
                # create trafo image
                self.trafo_image = numpy.ndarray(
                    (self.gwt.number_of_wavelets, image.shape[0], image.shape[1]),
                    numpy.complex128,
                )

            # perform Gabor wavelet transform
            self.gwt.transform(image, self.trafo_image)

            jet_length = self.gwt.number_of_wavelets * (2 if self.use_phases else 1)

            lgbphs_array = None
            # iterate through the layers of the trafo image
            for j in range(self.gwt.number_of_wavelets):
                # compute absolute part of complex response
                abs_image = numpy.abs(self.trafo_image[j])
                # Computes LBP histograms
                abs_blocks = bob.ip.base.lbphs(
                    abs_image, self.lbp, self.block_size, self.block_overlap
                )

                # Converts to Blitz array (of different dimensionalities)
                self.n_bins = abs_blocks.shape[1]
                self.n_blocks = abs_blocks.shape[0]

                if self.split is None:
                    shape = (self.n_blocks * self.n_bins * jet_length,)
                elif self.split == "blocks":
                    shape = (self.n_blocks, self.n_bins * jet_length)
                elif self.split == "wavelets":
                    shape = (jet_length, self.n_bins * self.n_blocks)
                elif self.split == "both":
                    shape = (jet_length * self.n_blocks, self.n_bins)
                else:
                    raise ValueError(
                        "The split parameter must be one of ['blocks', 'wavelets', 'both'] or None"
                    )

                # create new array if not done yet
                if lgbphs_array is None:
                    lgbphs_array = numpy.ndarray(shape, "float64")

                # fill the array with the absolute values of the Gabor wavelet transform
                self._fill(lgbphs_array, abs_blocks, j)

                if self.use_phases:
                    # compute phase part of complex response
                    phase_image = numpy.angle(self.trafo_image[j])
                    # Computes LBP histograms
                    phase_blocks = bob.ip.base.lbphs(
                        phase_image, self.lbp, self.block_size, self.block_overlap
                    )
                    # fill the array with the phases at the end of the blocks
                    self._fill(lgbphs_array, phase_blocks, j + self.gwt.number_of_wavelets)

            # return the concatenated list of all histograms
            return self._sparsify(lgbphs_array)

        return [_extract(x) for x in X]

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("lbp")
        d.pop("gwt")
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._init_non_pickables()

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}

    def fit(self, X, y=None):
        return self