import logging

from .Base import Base
from .croppers import FaceCropBoundingBox, FaceEyesNorm

logger = logging.getLogger("bob.bio.face")
from bob.bio.base import load_resource


class FaceCrop(Base):
    """

    Crops the face according to the given annotations.

    This class is designed to perform a geometric normalization of the face based
    on the eye locations, using :py:class:`bob.bio.face.preprocessor.croppers.FaceEyesNorm`. Usually,
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

    allow_upside_down_normalized_faces: bool, optional
      If ``False`` (default), a ValueError is raised when normalized faces are going to be
      upside down compared to input image. This allows you to catch wrong annotations in
      your database easily. If you are sure about your input, you can set this flag to
      ``True``.

    annotator : :any:`bob.bio.base.annotator.Annotator`
      If provided, the annotator will be used if the required annotations are
      missing.

    cropper:
        Pointer to a function that will crops using the annotations

    kwargs
      Remaining keyword parameters passed to the :py:class:`Base` constructor,
      such as ``color_channel`` or ``dtype``.
    """

    def __init__(
        self,
        cropped_image_size,
        cropped_positions=None,
        cropper=None,
        fixed_positions=None,
        annotator=None,
        allow_upside_down_normalized_faces=False,
        **kwargs,
    ):
        # call base class constructor
        Base.__init__(self, **kwargs)

        # Patching image size
        if isinstance(cropped_image_size, int):
            cropped_image_size = (cropped_image_size, cropped_image_size)

        # SEssion the cropper
        self.allow_upside_down_normalized_faces = (
            allow_upside_down_normalized_faces
        )
        if cropper is None:
            cropper = FaceEyesNorm(
                cropped_positions,
                cropped_image_size,
                allow_upside_down_normalized_faces=allow_upside_down_normalized_faces,
            )
        self.cropper = cropper

        # check parameters

        # copy parameters
        self.cropped_image_size = cropped_image_size
        self.cropped_positions = cropped_positions
        # self.cropped_keys = sorted(cropped_positions.keys())

        self.fixed_positions = fixed_positions
        if isinstance(annotator, str):
            annotator = load_resource(annotator, "annotator")
        self.annotator = annotator

        # create objects required for face cropping
        self.cropper = cropper

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

            # Priority to fixed position annotations
            if self.fixed_positions is not None:
                annot = self.fixed_positions

            # if annotations are missing and we don't have an annotator
            # return None.
            if annot is None and self.annotator is None:
                logger.warn(
                    "Cannot crop face without valid annotations or "
                    "fixed_positions or an annotator. Returning None. "
                    "The annotations were: {}".format(annot)
                )
                return None

            # convert to the desired color channel
            image = self.change_color_channel(image)

            # annotate the image if annotations are missing AND we don't have fixed_positions
            if annot is None and self.annotator is not None:
                annot = self.annotator([image], annotations=[annot])[0]
                if annot is None:
                    logger.warn(
                        "The annotator failed and the annot are missing too"
                        ". Returning None."
                    )
                    return None

            # crop face
            return self.data_type(self.cropper.transform(image, annot))

        if annotations is None:
            return [_crop(data, None) for data in X]
        else:
            return [_crop(data, annot) for data, annot in zip(X, annotations)]


class MultiFaceCrop(Base):
    """Wraps around FaceCrop to enable a dynamical cropper that can handle several annotation types.
    Initialization and usage is similar to the FaceCrop, but the main difference here is that one specifies
    a *list* of cropped_positions, and optionally a *list* of associated fixed positions.

    For each set of cropped_positions in the list, a new FaceCrop will be instantiated that handles this
    exact set of annotations.
    When calling the *transform* method, the MultiFaceCrop matches each sample to its associated cropper
    based on the received annotation, then performs the cropping of each subset, and finally gathers the results.

    If there is more than one cropper matching with the annotations, the **first valid** cropper will be taken.
    In case none of the croppers match with the received annotations, a ``ValueError`` is raised.

    Parameters
    ----------

        croppers_list : list
            A list of :py:class:`FaceCrop` that crops the face

    """

    def __init__(
        self,
        croppers_list,
    ):
        assert isinstance(croppers_list, list)
        for cl in croppers_list:
            assert isinstance(cl, FaceCrop)
        self.croppers_list = croppers_list

    def transform(self, X, annotations=None):

        # Assign each sample to its matching cropper
        transformed_samples = []
        for X_elem, annotations_elem in zip(X, annotations):
            cropped_sample = None
            for cropper in self.croppers_list:
                # Matching the first possible cropper that works
                try:
                    cropped_sample = cropper.transform(
                        [X_elem], [annotations_elem]
                    )[0]
                    break
                except Exception:
                    continue

            if cropped_sample is None:
                raise ValueError(
                    "No cropper found for annotations {}".format(
                        annotations_elem
                    )
                )

            transformed_samples.append(cropped_sample)

        # Gather the results
        return transformed_samples


class BoundingBoxAnnotatorCrop(Base):
    """
    This face cropper uses a 2 stage strategy to crop and align faces in case `annotation_type` has a bounding-box.
    In the first stage, it crops the face using the {`topleft`, `bottomright`} parameters and expands them using a `margin` factor.
    In the second stage, it uses the `annotator` to estimate {`leye` and `reye`} to make the crop using :py:class:`bob.bio.face.preprocessor.croppers.FaceEyesNorm`.
    In case the annotator doesn't work, it returns the cropped face using the `bounding-box` coordinates.

    .. warning::
            `cropped_positions` must be set with `leye`, `reye`, `topleft` and `bottomright` positions


    Parameters
    ----------

    eyes_cropper: :py:class:`bob.bio.face.preprocessor.croppers.FaceEyesNorm`
        This is the cropper that will be used to crop the face using eyes positions


    annotator : :any:`bob.bio.base.annotator.Annotator`
      This is the annotator that will be used to detect faces in the cropped images.


    """

    def __init__(
        self,
        eyes_cropper,
        annotator,
        margin=0.5,
    ):

        self.eyes_cropper = eyes_cropper
        self.margin = margin
        self.face_cropper = FaceCropBoundingBox(
            final_image_size=self.eyes_cropper.final_image_size, margin=margin
        )
        if isinstance(annotator, str):
            annotator = load_resource(annotator, "annotator")
        self.annotator = annotator

    def transform(self, X, annotations=None):
        """
        Crops the face using the two-stage croppers

        Parameters
        ----------

        X : list(numpy.ndarray)
            List of images to be cropped

        annotations : list(dict)
            Annotations for each image. Each annotation must contain the following keys:


        """

        faces = []

        for x, annot in zip(X, annotations):

            face_crop = self.face_cropper.transform(x, annot, resize=False)

            # get the coordinates with the annotator
            annotator_annotations = self.annotator([face_crop])[0]

            # If nothing was detected OR if the annotations are swaped, return the cropped face
            if (
                annotator_annotations is None
                or annotator_annotations["reye"][1]
                > annotator_annotations["leye"][1]
            ):
                logger.warning(
                    f"Unable to detect face in bounding box. Got : {annotator_annotations}. Cropping will be only based on bounding-box."
                )

                # append original image cropped with original bounding boxes
                faces.append(self.face_cropper.transform(x, annot, resize=True))
            else:

                faces.append(
                    self.eyes_cropper.transform(
                        face_crop, annotator_annotations
                    )
                )

        return faces
