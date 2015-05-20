import math
import numpy

import bob.ip.facedetect
import bob.ip.flandmark

import bob.ip.base
import numpy

from .Base import Base
from .utils import load_cropper_only
from bob.bio.base.preprocessor import Preprocessor


class FaceDetect (Base):

  def __init__(
      self,
      face_cropper,
      cascade = None,
      use_flandmark = False,
      detection_overlap = 0.2,
      distance = 2,
      scale_base = math.pow(2., -1./16.),
      lowest_scale = 0.125,
      mask_sigma = None,         # The sigma for random values areas outside image
      mask_neighbors = 5,        # The number of neighbors to consider while extrapolating
      mask_seed = None,          # The seed for generating random values during extrapolation
      **kwargs
  ):
    """Performs a face detection in the given image (ignoring any annotations)."""
    # call base class constructors
    Base.__init__(self, **kwargs)

    Preprocessor.__init__(
      self,
      face_cropper = face_cropper,
      cascade = cascade,
      use_flandmark = use_flandmark,
      detection_overlap = detection_overlap,
      distance = distance,
      scale_base = scale_base,
      lowest_scale = lowest_scale
    )

    self.sampler = bob.ip.facedetect.Sampler(scale_factor=scale_base, lowest_scale=lowest_scale, distance=distance)
    if cascade is None:
      self.cascade = bob.ip.facedetect.default_cascade()
    else:
      self.cascade = bob.ip.facedetect.Cascade(bob.io.base.HDF5File(cascade))
    self.detection_overlap = detection_overlap
    self.flandmark = bob.ip.flandmark.Flandmark() if use_flandmark else None
    self.quality = None

    self.cropper = load_cropper_only(face_cropper)


  def _landmarks(self, image, bounding_box):
    # get the landmarks in the face
    if self.flandmark is not None:
      # use the flandmark detector
      uint8_image = image.astype(numpy.uint8)
      # make the bounding box square shape by extending the horizontal position by 2 pixels times width/20
      bb = bob.ip.facedetect.BoundingBox(topleft = (bounding_box.top_f, bounding_box.left_f - bounding_box.size[1] / 10.), size = bounding_box.size)

      top = max(bb.top, 0)
      left = max(bb.left, 0)
      bottom = min(bb.bottom, image.shape[0])
      right = min(bb.right, image.shape[1])
      landmarks = self.flandmark.locate(uint8_image, top, left, bottom-top, right-left)

      if landmarks is not None and len(landmarks):
        return {
          'reye' : ((landmarks[1][0] + landmarks[5][0])/2., (landmarks[1][1] + landmarks[5][1])/2.),
          'leye' : ((landmarks[2][0] + landmarks[6][0])/2., (landmarks[2][1] + landmarks[6][1])/2.)
        }
      else:
        utils.warn("Could not detect landmarks -- using estimated landmarks")

    # estimate from default locations
    return bob.ip.facedetect.expected_eye_positions(bounding_box)


  def crop_face(self, image, annotations=None):
    # detect the face
    bounding_box, self.quality = bob.ip.facedetect.detect_single_face(image, self.cascade, self.sampler, self.detection_overlap)

    # get the eye landmarks
    annotations = self._landmarks(image, bounding_box)

    # apply face cropping
    return self.cropper.crop_face(image, annotations)



  def __call__(self, image, annotations=None):
    # convert to the desired color channel
    image = self.color_channel(image)

    # detect face and crop it
    image = self.crop_face(image)

    # convert data type
    return self.data_type(image)
