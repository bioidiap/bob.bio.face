import bob.ip.base
import numpy
import logging

from .Base import Base
from sklearn.base import TransformerMixin, BaseEstimator

logger = logging.getLogger("bob.bio.face")
from bob.bio.base import load_resource


class FaceCrop(Base):
    """Crops the face according to the given annotations.

    This class is designed to perform a geometric normalization of the face based
    on the eye locations, using :py:class:`bob.ip.base.FaceEyesNorm`. Usually,
    when executing the :py:meth:`crop_face` function, the image and the eye
    locations have to be specified. There, the given image will be transformed
    such that the eye locations will be placed at specific locations in the
    resulting image. These locations, as well as the size of the cropped image,
    need to be specified in the constructor of this class, as
    ``cropped_positions`` and ``cropped_image_size``.

    Some image databases do not provide eye locations, but rather bounding boxes.
    This is not a problem at all.
    Simply define the coordinates, where you want your ``cropped_positions`` to
    be in the cropped image, by specifying the same keys in the dictionary that
    will be given as ``annotations`` to the :py:meth:`crop_face` function.

    .. note::

      These locations can even be outside of the cropped image boundary, i.e.,
      when the crop should be smaller than the annotated bounding boxes.

    Sometimes, databases provide pre-cropped faces, where the eyes are located at
    (almost) the same position in all images. Usually, the cropping does not
    conform with the cropping that you like (i.e., image resolution is wrong, or
    too much background information). However, the database does not provide eye
    locations (since they are almost identical for all images). In that case, you
    can specify the ``fixed_positions`` in the constructor, which will be taken
    instead of the ``annotations`` inside the :py:meth:`crop_face` function (in
    which case the ``annotations`` are ignored).

    Sometimes, the crop of the face is outside of the original image boundaries.
    Usually, these pixels will simply be left black, resulting in sharp edges in
    the image. However, some feature extractors do not like these sharp edges. In
    this case, you can set the ``mask_sigma`` to copy pixels from the valid
    border of the image and add random noise (see
    :py:func:`bob.ip.base.extrapolate_mask`).


    Parameters
    ----------

    cropped_image_size : (int, int)
      The resolution of the cropped image, in order (HEIGHT,WIDTH); if not given,
      no face cropping will be performed

    cropped_positions : dict
      The coordinates in the cropped image, where the annotated points should be
      put to. This parameter is a dictionary with usually two elements, e.g.,
      ``{'reye':(RIGHT_EYE_Y, RIGHT_EYE_X) , 'leye':(LEFT_EYE_Y, LEFT_EYE_X)}``.
      However, also other parameters, such as ``{'topleft' : ..., 'bottomright' :
      ...}`` are supported, as long as the ``annotations`` in the `__call__`
      function are present.

    fixed_positions : dict or None
      If specified, ignore the annotations from the database and use these fixed
      positions throughout.

    mask_sigma : float or None
      Fill the area outside of image boundaries with random pixels from the
      border, by adding noise to the pixel values. To disable extrapolation, set
      this value to ``None``. To disable adding random noise, set it to a
      negative value or 0.

    mask_neighbors : int
      The number of neighbors used during mask extrapolation. See
      :py:func:`bob.ip.base.extrapolate_mask` for details.

    mask_seed : int or None
      The random seed to apply for mask extrapolation.

      .. warning::

         When run in parallel, the same random seed will be applied to all
         parallel processes. Hence, results of parallel execution will differ
         from the results in serial execution.

    allow_upside_down_normalized_faces: bool, optional
      If ``False`` (default), a ValueError is raised when normalized faces are going to be
      upside down compared to input image. This allows you to catch wrong annotations in
      your database easily. If you are sure about your input, you can set this flag to
      ``True``.

    annotator : :any:`bob.bio.base.annotator.Annotator`
      If provided, the annotator will be used if the required annotations are
      missing.

    kwargs
      Remaining keyword parameters passed to the :py:class:`Base` constructor,
      such as ``color_channel`` or ``dtype``.
    """

    def __init__(
        self,
        cropped_image_size,
        cropped_positions,
        fixed_positions=None,
        mask_sigma=None,
        mask_neighbors=5,
        mask_seed=None,
        annotator=None,
        allow_upside_down_normalized_faces=False,
        **kwargs,
    ):

        Base.__init__(self, **kwargs)

        if isinstance(cropped_image_size, int):
            cropped_image_size = (cropped_image_size, cropped_image_size)

        if isinstance(cropped_positions, str):
            face_size = cropped_image_size[0]

            if cropped_positions == "eyes-center":
                eyes_distance = (face_size + 1) / 2.0
                eyes_center = (face_size / 4.0, (face_size - 0.5) / 2.0)
                right_eye = (eyes_center[0], eyes_center[1] - eyes_distance / 2)
                left_eye = (eyes_center[0], eyes_center[1] + eyes_distance / 2)
                cropped_positions = {"reye": right_eye, "leye": left_eye}

            elif cropped_positions == "bounding-box":
                cropped_positions = {
                    "topleft": (0, 0),
                    "bottomright": cropped_image_size,
                }

            else:
                raise ValueError(
                    f"Got {cropped_positions} as cropped_positions "
                    "while only eyes and bbox strings are supported."
                )

        # call base class constructor
        self.cropped_image_size = cropped_image_size
        self.cropped_positions = cropped_positions
        self.fixed_positions = fixed_positions
        self.mask_sigma = mask_sigma
        self.mask_neighbors = mask_neighbors
        self.mask_seed = mask_seed

        # check parameters
        assert len(cropped_positions) == 2
        if fixed_positions:
            assert len(fixed_positions) == 2

        # copy parameters
        self.cropped_image_size = cropped_image_size
        self.cropped_positions = cropped_positions
        self.cropped_keys = sorted(cropped_positions.keys())
        self.fixed_positions = fixed_positions
        self.mask_sigma = mask_sigma
        self.mask_neighbors = mask_neighbors
        self.mask_seed = mask_seed
        if isinstance(annotator, str):
            annotator = load_resource(annotator, "annotator")
        self.annotator = annotator
        self.allow_upside_down_normalized_faces = allow_upside_down_normalized_faces

        # create objects required for face cropping
        self.cropper = bob.ip.base.FaceEyesNorm(
            crop_size=cropped_image_size,
            right_eye=cropped_positions[self.cropped_keys[0]],
            left_eye=cropped_positions[self.cropped_keys[1]],
        )
        self.cropped_mask = numpy.ndarray(cropped_image_size, numpy.bool)

        self._init_non_pickables()

    def _init_non_pickables(self):
        self.mask_rng = (
            bob.core.random.mt19937(self.mask_seed)
            if self.mask_seed is not None
            else bob.core.random.mt19937()
        )
        self.cropper = bob.ip.base.FaceEyesNorm(
            crop_size=self.cropped_image_size,
            right_eye=self.cropped_positions[self.cropped_keys[0]],
            left_eye=self.cropped_positions[self.cropped_keys[1]],
        )

    def crop_face(self, image, annotations=None):
        """Crops the face.
        Executes the face cropping on the given image and returns the cropped
        version of it.

        Parameters
        ----------
        image : 2D :py:class:`numpy.ndarray`
            The face image to be processed.

        annotations : dict or ``None``
            The annotations that fit to the given image. ``None`` is only accepted,
            when ``fixed_positions`` were specified in the constructor.

        Returns
        -------
        face : 2D :py:class:`numpy.ndarray` (float)
            The cropped face.

        Raises
        ------
        ValueError
            If the annotations is None.
        """
        if self.fixed_positions is not None:
            annotations = self.fixed_positions
        if annotations is None:
            raise ValueError(
                "Cannot perform image cropping since annotations are not given, and "
                "no fixed annotations are specified."
            )

        assert isinstance(annotations, dict)
        if not all(k in annotations for k in self.cropped_keys):
            raise ValueError(
                "At least one of the expected annotations '%s' are not given "
                "in '%s'." % (self.cropped_keys, annotations.keys())
            )

        reye = self.cropped_keys[0]
        leye = self.cropped_keys[1]
        reye_desired_width = self.cropped_positions[reye][1]
        leye_desired_width = self.cropped_positions[leye][1]
        right_eye = annotations[reye]
        left_eye = annotations[leye]
        if not self.allow_upside_down_normalized_faces:
            if (
                reye_desired_width > leye_desired_width and right_eye[1] < left_eye[1]
            ) or (
                reye_desired_width < leye_desired_width and right_eye[1] > left_eye[1]
            ):
                raise ValueError(
                    "Looks like {leye} and {reye} in annotations: {annot} are swapped. "
                    "This will make the normalized face upside down (compared to the original "
                    "image). Most probably your annotations are wrong. Otherwise, you can set "
                    "the ``allow_upside_down_normalized_faces`` parameter to "
                    "True.".format(leye=leye, reye=reye, annot=annotations)
                )

        # create output
        mask = numpy.ones(image.shape[-2:], dtype=numpy.bool)
        shape = (
            self.cropped_image_size
            if image.ndim == 2
            else [image.shape[0]] + list(self.cropped_image_size)
        )
        cropped_image = numpy.zeros(shape)
        self.cropped_mask[:] = False

        # perform the cropping
        self.cropper(
            image,  # input image
            mask,  # full input mask
            cropped_image,  # cropped image
            self.cropped_mask,  # cropped mask
            # position of first annotation, usually right eye
            right_eye=right_eye,
            # position of second annotation, usually left eye
            left_eye=left_eye,
        )

        if self.mask_sigma is not None:
            # extrapolate the mask so that pixels outside of the image original image
            # region are filled with border pixels
            if cropped_image.ndim == 2:
                bob.ip.base.extrapolate_mask(
                    self.cropped_mask,
                    cropped_image,
                    self.mask_sigma,
                    self.mask_neighbors,
                    self.mask_rng,
                )
            else:
                [
                    bob.ip.base.extrapolate_mask(
                        self.cropped_mask,
                        cropped_image_channel,
                        self.mask_sigma,
                        self.mask_neighbors,
                        self.mask_rng,
                    )
                    for cropped_image_channel in cropped_image
                ]

        return cropped_image

    def is_annotations_valid(self, annotations):
        if not annotations:
            return False
        # check if the required keys are available
        return all(key in annotations for key in self.cropped_keys)

    def transform(self, X, annotations=None):
        """Aligns the given image according to the given annotations.

        First, the desired color channel is extracted from the given image.
        Afterward, the face is cropped, according to the given ``annotations`` (or
        to ``fixed_positions``, see :py:meth:`crop_face`). Finally, the resulting
        face is converted to the desired data type.

        Parameters
        ----------
        image : 2D or 3D :py:class:`numpy.ndarray`
            The face image to be processed.
        annotations : dict or ``None``
            The annotations that fit to the given image.

        Returns
        -------
        face : 2D :py:class:`numpy.ndarray`
            The cropped face.
        """

        def _crop(image, annot):
            # if annotations are missing and cannot do anything else return None.
            if (
                not self.is_annotations_valid(annot)
                and not self.fixed_positions
                and self.annotator is None
            ):
                logger.warn(
                    "Cannot crop face without valid annotations or "
                    "fixed_positions or an annotator. Returning None. "
                    "The annotations were: {}".format(annot)
                )
                return None

            # convert to the desired color channel
            image = self.change_color_channel(image)

            # annotate the image if annotations are missing
            if (
                not self.is_annotations_valid(annot)
                and not self.fixed_positions
                and self.annotator is not None
            ):
                annot = self.annotator([image], annotations=[annot])[0]
                if not self.is_annotations_valid(annot):
                    logger.warn(
                        "The annotator failed and the annot are missing too"
                        ". Returning None."
                    )
                    return None

            # crop face
            return self.data_type(self.crop_face(image, annot))

        if annotations is None:
            return [_crop(data, None) for data in X]
        else:
            return [_crop(data, annot) for data, annot in zip(X, annotations)]

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("mask_rng")
        d.pop("cropper")
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._init_non_pickables()


class MultiFaceCrop(Base):
    """ Wraps around FaceCrop to enable a dynamical cropper that can handle several annotation types.
    Initialization and usage is similar to the FaceCrop, but the main difference here is that one specifies
    a *list* of cropped_positions, and optionally a *list* of associated fixed positions.

    For each set of cropped_positions in the list, a new FaceCrop will be instanciated that handles this
    exact set of annotations.
    When calling the *transform* method, the MultiFaceCrop matches each sample to its associated cropper
    based on the received annotation, then performs the cropping of each subset, and finally gathers the results.

    In case of ambiguity (when no cropper is a match for the received annotations, or when several croppers
    match the received annotations), raises a ValueError.

    """

    def __init__(
        self,
        cropped_image_size,
        cropped_positions_list,
        fixed_positions_list=None,
        mask_sigma=None,
        mask_neighbors=5,
        mask_seed=None,
        annotator=None,
        allow_upside_down_normalized_faces=False,
        **kwargs,
    ):

        assert isinstance(cropped_positions_list, list)
        if fixed_positions_list is None:
            fixed_positions_list = [None] * len(cropped_positions_list)
        assert isinstance(fixed_positions_list, list)

        self.croppers = {}
        for cropped_positions, fixed_positions in zip(
            cropped_positions_list, fixed_positions_list
        ):
            assert len(cropped_positions) == 2
            self.croppers[tuple(cropped_positions)] = FaceCrop(
                cropped_image_size,
                cropped_positions,
                fixed_positions,
                mask_sigma,
                mask_neighbors,
                mask_seed,
                annotator,
                allow_upside_down_normalized_faces,
                **kwargs,
            )

    def transform(self, X, annotations=None):
        subsets = {k: {"X": [], "annotations": []} for k in self.croppers.keys()}

        def assign(X_elem, annotations_elem):
            # Assign a single sample to its matching cropper

            # Compare the received annotations keys to the cropped_positions keys of each cropper
            valid_keys = [
                k
                for k in self.croppers.keys()
                if set(k).issubset(set(annotations_elem.keys()))
            ]

            # Ensure exactly one cropper is a match
            if len(valid_keys) != 1:
                raise ValueError(
                    "Cropper selection from the annotations is ambiguous ({} valid croppers)".format(
                        len(valid_keys)
                    )
                )
            else:
                # Assign the sample to this particuler cropper
                cropper_key = valid_keys[0]
                subsets[cropper_key]["X"].append(X_elem)
                subsets[cropper_key]["annotations"].append(annotations_elem)

        # Assign each sample to its matching cropper
        for X_elem, annotations_elem in zip(X, annotations):
            assign(X_elem, annotations_elem)

        # Call each FaceCrop on its sample subset
        transformed_subsets = {
            k: self.croppers[k].transform(**subsets[k]) for k in subsets.keys()
        }

        # Gather the results
        return [item for sublist in transformed_subsets.values() for item in sublist]

    def fit(self, X, y=None):
        return self
