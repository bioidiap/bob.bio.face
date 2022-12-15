#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import cv2
import numpy
import numpy as np

from .Base import Base
from .utils import load_cropper


class HistogramEqualization(Base):
    """Crops the face (if desired) and performs histogram equalization to photometrically enhance the image.

    Parameters
    ----------

    face_cropper : str or :py:class:`bob.bio.face.preprocessor.FaceCrop` or :py:class:`bob.bio.face.preprocessor.FaceDetect` or ``None``
      The face image cropper that should be applied to the image.
      If ``None`` is selected, no face cropping is performed.
      Otherwise, the face cropper might be specified as a registered resource, a configuration file, or an instance of a preprocessor.

      .. note:: The given class needs to contain a ``crop_face`` method.

    kwargs
      Remaining keyword parameters passed to the :py:class:`Base` constructor, such as ``color_channel`` or ``dtype``.
    """

    def __init__(self, face_cropper, **kwargs):

        Base.__init__(self, **kwargs)

        self.face_cropper = (face_cropper,)
        self.cropper = load_cropper(face_cropper)

    def equalize_histogram(self, image):
        """equalize_histogram(image) -> equalized

        Performs the histogram equalization on the given image.

        Parameters
        ----------

        image : 2D :py:class:`numpy.ndarray`
          The image to berform histogram equalization with.
          The image will be transformed to type ``uint8`` before computing the histogram.

        Returns
        -------

        equalized : 2D :py:class:`numpy.ndarray` (float)
          The photometrically enhanced image.
        """
        return cv2.equalizeHist(np.round(image).astype(numpy.uint8))

    def transform(self, X, annotations=None):
        """
        Aligns the given image according to the given annotations.

        First, the desired color channel is extracted from the given image.
        Afterward, the face is eventually cropped using the ``face_cropper`` specified in the constructor.
        Then, the image is photometrically enhanced using histogram equalization.
        Finally, the resulting face is converted to the desired data type.

        Parameters
        ----------

        X : 2D or 3D :py:class:`numpy.ndarray`
          The face image to be processed.

        annotations : dict or ``None``
          The annotations that fit to the given image.
          Might be ``None``, when the ``face_cropper`` is ``None`` or of type :py:class:`FaceDetect`.

        Returns
        -------

        face : 2D :py:class:`numpy.ndarray`
          The cropped and photometrically enhanced face.
        """

        def _crop(image, annotations):
            image = self.change_color_channel(image)
            if self.cropper is not None:
                # TODO: USE THE TAG `ALLOW_ANNOTATIONS`
                image = (
                    self.cropper.transform([image])
                    if annotations is None
                    else self.cropper.transform([image], [annotations])
                )
                image = self.equalize_histogram(image[0])
            else:
                # Handle with the cropper is None
                image = self.equalize_histogram(image)

            return self.data_type(image)

        if annotations is None:
            return [_crop(data) for data in X]
        else:
            return [_crop(data, annot) for data, annot in zip(X, annotations)]
