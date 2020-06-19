import math
import numpy

import bob.ip.facedetect
import bob.ip.flandmark

import bob.ip.base
import numpy

from .Base import Base
from .utils import load_cropper_only
from sklearn.utils import check_array
from bob.pipelines.sample import SampleBatch
import logging

logger = logging.getLogger("bob.bio.face")


class FaceDetect(Base):
    """Performs a face detection (and facial landmark localization) in the given image and crops the face.

  This class is designed to perform a geometric normalization of the face based on the detected face.
  Face detection is performed using :ref:`bob.ip.facedetect <bob.ip.facedetect>`.
  Particularly, the function :py:func:`bob.ip.facedetect.detect_single_face` is executed, which will *always* return *exactly one* bounding box, even if the image contains more than one face, or no face at all.
  The speed of the face detector can be regulated using the ``cascade``, ``distance` ``scale_base`` and ``lowest_scale`` parameters.
  The number of overlapping detected bounding boxes that should be joined can be selected by ``detection_overlap``.
  Please see the documentation of :ref:`bob.ip.facedetect <bob.ip.facedetect>` for more details about these parameters.

  Additionally, facial landmarks can be detected using the :ref:`bob.ip.flandmark`.
  If enabled using ``use_flandmark = True`` in the constructor, it is tried to obtain the facial landmarks inside the detected facial area.
  If landmarks are found, these are used to geometrically normalize the face.
  Otherwise, the eye locations are estimated based on the bounding box.
  This is also applied, when ``use_flandmark = False.``

  The face cropping itself is done by the given ``face_cropper``.
  This cropper can either be an instance of :py:class:`FaceCrop` (or any other class that provides a similar ``crop_face`` function), or it can be the resource name of a face cropper, such as ``'face-crop-eyes'``.

  **Parameters:**

  face_cropper : :py:class:`bob.bio.face.preprocessor.FaceCrop` or str
    The face cropper to be used to crop the detected face.
    Might be an instance of a :py:class:`FaceCrop` or the name of a face cropper resource.

  cascade : str or ``None``
    The file name, where a face detector cascade can be found.
    If ``None``, the default cascade for frontal faces :py:func:`bob.ip.facedetect.default_cascade` is used.

  use_flandmark : bool
    If selected, :py:class:`bob.ip.flandmark.Flandmark` is used to detect the eye locations.
    Otherwise, the eye locations are estimated based on the detected bounding box.

  detection_overlap : float
    See :py:func:`bob.ip.facedetect.detect_single_face`.

  distance : int
    See the Sampling section in the :ref:`Users Guide of bob.ip.facedetect <bob.ip.facedetect>`.

  scale_base : float
    See the Sampling section in the :ref:`Users Guide of bob.ip.facedetect <bob.ip.facedetect>`.

  lowest_scale : float
    See the Sampling section in the :ref:`Users Guide of bob.ip.facedetect <bob.ip.facedetect>`.

  kwargs
    Remaining keyword parameters passed to the :py:class:`Base` constructor, such as ``color_channel`` or ``dtype``.
  """

    def __init__(
        self,
        face_cropper,
        cascade=None,
        use_flandmark=False,
        detection_overlap=0.2,
        distance=2,
        scale_base=math.pow(2.0, -1.0 / 16.0),
        lowest_scale=0.125,
        **kwargs
    ):
        # call base class constructors
        Base.__init__(self, **kwargs)

        self.face_cropper = face_cropper
        self.cascade=cascade
        self.use_flandmark=use_flandmark
        self.detection_overlap=detection_overlap
        self.distance=distance
        self.scale_base=scale_base
        self.lowest_scale=lowest_scale

        assert face_cropper is not None

        self.scale_base = scale_base
        self.lowest_scale = lowest_scale
        self.distance = distance
        self.cascade = cascade
        self.use_flandmark = use_flandmark

        self.detection_overlap = detection_overlap
        self.quality = None

        self.cropper = load_cropper_only(face_cropper)

        self._init_non_pickables()

    def _init_non_pickables(self):
        self.sampler = bob.ip.facedetect.Sampler(
            scale_factor=self.scale_base,
            lowest_scale=self.lowest_scale,
            distance=self.distance,
        )

        if self.cascade is None:
            self.cascade_classifier = bob.ip.facedetect.default_cascade()
        else:
            self.cascade_classifier = bob.ip.facedetect.Cascade(
                bob.io.base.HDF5File(self.cascade)
            )

        self.flandmark = bob.ip.flandmark.Flandmark() if self.use_flandmark else None

    def _landmarks(self, image, bounding_box):
        """Try to detect the landmarks in the given bounding box, and return the eye locations."""
        # get the landmarks in the face
        if self.flandmark is not None:
            # use the flandmark detector

            # make the bounding box square shape by extending the horizontal position by 2 pixels times width/20
            bb = bob.ip.facedetect.BoundingBox(
                topleft=(
                    bounding_box.top_f,
                    bounding_box.left_f - bounding_box.size[1] / 10.0,
                ),
                size=bounding_box.size,
            )

            top = max(bb.top, 0)
            left = max(bb.left, 0)
            bottom = min(bb.bottom, image.shape[0])
            right = min(bb.right, image.shape[1])
            landmarks = self.flandmark.locate(
                image, top, left, bottom - top, right - left
            )

            if landmarks is not None and len(landmarks):
                return {
                    "reye": (
                        (landmarks[1][0] + landmarks[5][0]) / 2.0,
                        (landmarks[1][1] + landmarks[5][1]) / 2.0,
                    ),
                    "leye": (
                        (landmarks[2][0] + landmarks[6][0]) / 2.0,
                        (landmarks[2][1] + landmarks[6][1]) / 2.0,
                    ),
                }
            else:
                logger.warn("Could not detect landmarks -- using estimated landmarks")

        # estimate from default locations
        return bob.ip.facedetect.expected_eye_positions(bounding_box)

    def crop_face(self, image, annotations=None):
        """crop_face(image, annotations = None) -> face

    Detects the face (and facial landmarks), and used the ``face_cropper`` given in the constructor to crop the face.

    **Parameters:**

    image : 2D or 3D :py:class:`numpy.ndarray`
      The face image to be processed.

    annotations : any
      Ignored.

    **Returns:**

    face : 2D or 3D :py:class:`numpy.ndarray` (float)
      The detected and cropped face.
    """
        uint8_image = image.astype(numpy.uint8)
        if uint8_image.ndim == 3:
            uint8_image = bob.ip.color.rgb_to_gray(uint8_image)

        # detect the face
        bounding_box, self.quality = bob.ip.facedetect.detect_single_face(
            uint8_image, self.cascade_classifier, self.sampler, self.detection_overlap
        )

        # get the eye landmarks
        annotations = self._landmarks(uint8_image, bounding_box)

        # apply face cropping
        return self.cropper.crop_face(image, annotations)

    def transform(self, X, annotations=None):
        """__call__(image, annotations = None) -> face

    Aligns the given image according to the detected face bounding box or the detected facial features.

    First, the desired color channel is extracted from the given image.
    Afterward, the face is detected and cropped, see :py:meth:`crop_face`.
    Finally, the resulting face is converted to the desired data type.

    **Parameters:**

    image : 2D or 3D :py:class:`numpy.ndarray`
      The face image to be processed.

    annotations : any
      Ignored.

    **Returns:**

    face : 2D :py:class:`numpy.ndarray`
      The cropped face.
    """
        def _crop(image, annotation):

            # convert to the desired color channel
            image = self.color_channel(image)

            # detect face and crop it
            image = self.crop_face(image)

            # convert data type
            return self.data_type(image)


        if isinstance(X, SampleBatch):

            if annotations is None:
                return [_crop(data) for data in X]
            else:
                return [_crop(data, annot) for data, annot in zip(X, annotations)]

        else:
            return _crop(X, annotations)


    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop("sampler")
        d.pop("cascade_classifier")
        d.pop("flandmark")
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._init_non_pickables()
