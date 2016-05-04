import numpy
import bob.io.image
import bob.ip.color

from bob.bio.base.preprocessor import Preprocessor

class Base (Preprocessor):
  """Performs color space adaptations and data type corrections for the given image.

  **Parameters:**

  dtype : :py:class:`numpy.dtype` or convertible or ``None``
    The data type that the resulting image will have.

  color_channel : one of ``('gray', 'red', 'gren', 'blue', 'rgb')``
    The specific color channel, which should be extracted from the image.
  """

  def __init__(self, dtype = None, color_channel = 'gray'):
    Preprocessor.__init__(self, dtype=str(dtype), color_channel=color_channel)
    self.channel = color_channel
    self.dtype = dtype


  def color_channel(self, image):
    """color_channel(image) -> channel

    Returns the channel of the given image, which was selected in the constructor.
    Currently, gray, red, green and blue channels are supported.

    **Parameters:**

    image : 2D or 3D :py:class:`numpy.ndarray`
      The image to get the specified channel from.

    **Returns:**

    channel : 2D or 3D :py:class:`numpy.ndarray`
      The extracted color channel.
    """
    if image.ndim == 2:
      if self.channel == 'rgb':
        return bob.ip.color.gray_to_rgb(image)
      if self.channel != 'gray':
        raise ValueError("There is no rule to extract a " + channel + " image from a gray level image!")
      return image

    if self.channel == 'rgb':
      return image
    if self.channel == 'gray':
      return bob.ip.color.rgb_to_gray(image)
    if self.channel == 'red':
      return image[0,:,:]
    if self.channel == 'green':
      return image[1,:,:]
    if self.channel == 'blue':
      return image[2,:,:]

    raise ValueError("The image channel '%s' is not known or not yet implemented", self.channel)


  def data_type(self, image):
    """data_type(image) -> image

    Converts the given image into the data type specified in the constructor of this class.
    If no data type was specified, or the ``image`` is ``None``, no conversion is performed.

    **Parameters:**

    image : 2D or 3D :py:class:`numpy.ndarray`
      The image to convert.

    **Returns:**

    image : 2D or 3D :py:class:`numpy.ndarray`
      The image converted to the desired data type, if any.
    """
    if self.dtype is not None and image is not None:
      image = image.astype(self.dtype)
    return image


  def __call__(self, image, annotations = None):
    """__call__(image, annotations = None) -> image

    Extracts the desired color channel and converts to the desired data type.

    **Parameters:**

    image : 2D or 3D :py:class:`numpy.ndarray`
      The image to preprocess.

    annotations : any
      Ignored.

    **Returns:**

    image : 2D :py:class:`numpy.ndarray`
      The image converted converted to the desired color channel and type.
    """
    assert isinstance(image, numpy.ndarray) and image.ndim in (2,3)
    # convert to grayscale
    image = self.color_channel(image)
    return self.data_type(image)
