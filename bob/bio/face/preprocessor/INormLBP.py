#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import skimage

from .Base import Base
from .utils import load_cropper


class INormLBP(Base):
    """
    Performs I-Norm LBP on the given image.

    The supported LBP methods are the ones available on.
    https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.local_binary_pattern



    Parameters
    ----------

    face_cropper : str or :py:class:`bob.bio.face.preprocessor.FaceCrop` or :py:class:`bob.bio.face.preprocessor.FaceDetect` or ``None``
      The face image cropper that should be applied to the image.
      It might be specified as a registered resource, a configuration file, or an instance of a preprocessor.

      .. note:: The given class needs to contain a ``crop_face`` method.

    neighbors : int
        Number of circularly symmetric neighbor set points (quantization of the angular space)

    radius : int
      The radius of the LBP features to extract

    method : str
      The type of LBP to use. https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.local_binary_pattern


    """

    def __init__(
        self,
        face_cropper,
        neighbors=8,
        radius=2,  # Radius of the LBP
        method="default",  # Type of LBP
        **kwargs
    ):

        # call base class constructors
        Base.__init__(self, **kwargs)

        self.face_cropper = face_cropper
        self.radius = radius
        self.neighbors = neighbors
        self.method = method
        self.cropper = load_cropper(face_cropper)

    def transform(self, X, annotations=None):
        """__call__(image, annotations = None) -> face

        Aligns the given image according to the given annotations.

        First, the desired color channel is extracted from the given image.
        Afterward, the face is eventually cropped using the ``face_cropper`` specified in the constructor.
        Then, the image is photometrically enhanced by extracting LBP features [HRM06]_.
        Finally, the resulting face is converted to the desired data type.

        **Parameters:**

        image : 2D or 3D :py:class:`numpy.ndarray`
          The face image to be processed.

        annotations : dict or ``None``
          The annotations that fit to the given image.
          Might be ``None``, when the ``face_cropper`` is ``None`` or of type :py:class:`FaceDetect`.

        **Returns:**

        face : 2D :py:class:`numpy.ndarray`
          The cropped and photometrically enhanced face.
        """

        def _crop_one_sample(image, annotations=None):

            if self.cropper is not None:
                # TODO: USE THE TAG `ALLOW_ANNOTATIONS`
                image = (
                    self.cropper.transform([image])
                    if annotations is None
                    else self.cropper.transform([image], [annotations])
                )
                # LBP's doesn't work with batches, so we have to work this out
                # Also, we change the color channel *after* cropping : some croppers use MTCNN internally, that works on multichannel images
                image = self.change_color_channel(image[0])
                image = skimage.feature.local_binary_pattern(
                    image, P=self.neighbors, R=self.radius, method=self.method
                )
            else:
                # Handle with the cropper is None
                image = self.change_color_channel(image)
                # image = self.lbp_extractor(image)
                image = skimage.feature.local_binary_pattern(
                    image, P=self.neighbors, R=self.radius, method=self.method
                )

            return self.data_type(image)

        if annotations is None:
            return [_crop_one_sample(data) for data in X]
        else:
            return [
                _crop_one_sample(data, annot)
                for data, annot in zip(X, annotations)
            ]
