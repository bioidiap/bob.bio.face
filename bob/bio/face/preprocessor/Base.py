import numpy

from sklearn.base import BaseEstimator, TransformerMixin

from bob.bio.face.color import gray_to_rgb, rgb_to_gray


def change_color_channel(image, color_channel):
    if image.ndim == 2:
        if color_channel == "rgb":
            return gray_to_rgb(image)
        if color_channel != "gray":
            raise ValueError(
                "There is no rule to extract a "
                + color_channel
                + " image from a gray level image!"
            )
        return image

    if color_channel == "rgb":
        return image
    if color_channel == "bgr":
        return image[[2, 1, 0], ...]
    if color_channel == "gray":
        return rgb_to_gray(image)
    if color_channel == "red":
        return image[0, :, :]
    if color_channel == "green":
        return image[1, :, :]
    if color_channel == "blue":
        return image[2, :, :]

    raise ValueError(
        "The image channel '%s' is not known or not yet implemented",
        color_channel,
    )


class Base(TransformerMixin, BaseEstimator):
    """Performs color space adaptations and data type corrections for the given
    image.

    **Parameters:**

    dtype : :py:class:`numpy.dtype` or convertible or ``None``
      The data type that the resulting image will have.

    color_channel : one of ``('gray', 'red', 'gren', 'blue', 'rgb')``
      The specific color channel, which should be extracted from the image.
    """

    def __init__(self, dtype=None, color_channel="gray", **kwargs):
        self.color_channel = color_channel
        self.dtype = dtype

    @property
    def channel(self):
        return self.color_channel

    def _more_tags(self):
        return {"requires_fit": False}

    def fit(self, X, y=None):
        return self

    def change_color_channel(self, image):
        """color_channel(image) -> channel

        Returns the channel of the given image, which was selected in the
        constructor. Currently, gray, red, green and blue channels are supported.

        **Parameters:**

        image : 2D or 3D :py:class:`numpy.ndarray`
          The image to get the specified channel from.

        **Returns:**

        channel : 2D or 3D :py:class:`numpy.ndarray`
          The extracted color channel.
        """

        return change_color_channel(image, self.color_channel)

    def data_type(self, image):
        """
        Converts the given image into the data type specified in the constructor of
        this class. If no data type was specified, or the ``image`` is ``None``, no
        conversion is performed.

        Parameters
        ----------

        image : 2D or 3D :py:class:`numpy.ndarray`
          The image to convert.

        Returns
        -------

        image : 2D or 3D :py:class:`numpy.ndarray`
          The image converted to the desired data type, if any.
        """
        if self.dtype is not None and image is not None:
            image = image.astype(self.dtype)
        return image

    def transform(self, images, annotations=None):
        """
        Extracts the desired color channel and converts to the desired data type.

        Parameters
        ----------

        image : 2D or 3D :py:class:`numpy.ndarray`
          The image to preprocess.

        annotations : any
          Ignored.

        Returns
        -------

        image : 2D :py:class:`numpy.ndarray`
          The image converted converted to the desired color channel and type.
        """
        return [self._transform_one_image(img) for img in images]

    def _transform_one_image(self, image):
        assert isinstance(image, numpy.ndarray) and image.ndim in (2, 3)
        # convert to grayscale
        image = self.change_color_channel(image)
        return self.data_type(image)
