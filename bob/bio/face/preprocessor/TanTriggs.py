#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


import cv2
import numpy as np

from .Base import Base
from .utils import load_cropper


def compute_tan_triggs(
    image, gamma=0.2, sigma_0=1, sigma_1=2, size=11, threshold=10, alpha=0.1
):
    """
    Applies Tan&Triggs algorithm [TT10]_ to photometrically enhance the image


    Parameters
    ----------

    image : 2D numpy.ndarray
      The image to be processed.

    gamma : float
      [default: 0.2] The value of gamma for the gamma correction

    sigma_0 : float
      [default: 1] The standard deviation of the first Gaussian kernel used in the DoG filter to smooth the image.

    sigma_1 : float
      [default: 2] The standard deviation of the second Gaussian kernel used in the DoG filter to smooth the image.

    size : int
      [default: 11] The size of the Gaussian kernel used in the DoG filter to smooth the image.

    threshold : float
      [default: 10] The threshold used for the contrast equalization

    alpha : float
      [default: 0.1] The alpha value used for the contrast equalization


    """
    assert image.ndim == 2, "The image must be a 2D numpy.ndarray"

    # 1. Gamma correction
    gamma_image = np.power(image, gamma)

    # 2. DoG filter
    dog_1 = cv2.GaussianBlur(gamma_image, (size, size), sigma_0)
    dog_2 = cv2.GaussianBlur(gamma_image, (size, size), sigma_1)
    dog_image = dog_1 - dog_2

    # 3. Contrast equalization
    # first step - I:=I/mean(abs(I)^a)^(1/a)
    norm_fact = np.mean(np.abs(dog_image) ** alpha) ** (1 / alpha)
    dog_image /= norm_fact

    # second step - I:=I/mean(min(threshold,abs(I))^a)^(1/a)
    norm_fact = np.mean(np.minimum(threshold, np.abs(dog_image)) ** alpha) ** (
        1 / alpha
    )
    dog_image /= norm_fact

    # 4. I:= threshold * tanh( I / threshold )
    dog_image = np.tanh(dog_image / threshold) * threshold

    return dog_image


class TanTriggs(Base):
    """Crops the face (if desired) and applies Tan&Triggs algorithm [TT10]_ to photometrically enhance the image.

    Parameters
    ----------

    face_cropper : str or :py:class:`bob.bio.face.preprocessor.FaceCrop` or :py:class:`bob.bio.face.preprocessor.FaceDetect` or ``None``
      The face image cropper that should be applied to the image.
      If ``None`` is selected, no face cropping is performed.
      Otherwise, the face cropper might be specified as a registered resource, a configuration file, or an instance of a preprocessor.

      .. note:: The given class needs to contain a ``crop_face`` method.

    gamma, sigma0, sigma1, size, threshold, alpha
      Please refer to the [TT10]_ original paper.

    kwargs
      Remaining keyword parameters passed to the :py:class:`Base` constructor, such as ``color_channel`` or ``dtype``.
    """

    def __init__(
        self,
        face_cropper,
        gamma=0.2,
        sigma0=1,
        sigma1=2,
        size=5,
        threshold=10.0,
        alpha=0.1,
        **kwargs
    ):

        Base.__init__(self, **kwargs)

        # call base class constructor with its set of parameters

        self.face_cropper = face_cropper
        self.gamma = gamma
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.size = size
        self.threshold = threshold
        self.alpha = alpha

        self.gamma = gamma
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.size = size
        self.threshold = threshold
        self.alpha = alpha

        self.cropper = load_cropper(face_cropper)

    def transform(self, X, annotations=None):
        """__call__(image, annotations = None) -> face

        Aligns the given image according to the given annotations.

        First, the desired color channel is extracted from the given image.
        Afterward, the face is eventually cropped using the ``face_cropper`` specified in the constructor.
        Then, the image is photometrically enhanced using the Tan&Triggs algorithm [TT10]_.
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
                # We change the color channel *after* cropping : some croppers use MTCNN internally, that works on multichannel images
                image = self.change_color_channel(image[0])

                image = compute_tan_triggs(
                    image,
                    self.gamma,
                    self.sigma0,
                    self.sigma1,
                    self.size,
                    self.threshold,
                    self.alpha,
                )

            else:
                # Handle with the cropper is None
                image = self.change_color_channel(image)
                image = compute_tan_triggs(
                    image,
                    self.gamma,
                    self.sigma0,
                    self.sigma1,
                    self.size,
                    self.threshold,
                    self.alpha,
                )

            return self.data_type(image)

        if annotations is None:
            return [_crop_one_sample(data) for data in X]
        else:
            return [
                _crop_one_sample(data, annot)
                for data, annot in zip(X, annotations)
            ]
