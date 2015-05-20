import numpy
import bob.io.image
import bob.ip.color

from bob.bio.base.preprocessor import Preprocessor

class Base (Preprocessor):
  """Performs color space adaptations and data type corrections for the given image"""

  def __init__(self, dtype = None, color_channel = 'gray'):
    """Parameters of the constructor of this preprocessor:

    dtype : :py:class:`numpy.dtype` or convertible or ``None``
      The data type that the resulting image will have

    color_channel : one of ``('gray', 'red', 'gren', 'blue')`` or ``None``
      The specific color channel, which should be extracted from the image
    """
    Preprocessor.__init__(self, dtype=str(dtype), color_channel=color_channel)
    self.channel = color_channel
    self.dtype = dtype


  def color_channel(self, image):
    """Returns the desired channel of the given image. Currently, gray, red, green and blue channels are supported."""
    if self.channel is None:
      return image

    if image.ndim == 2:
      if self.channel != 'gray':
        raise ValueError("There is no rule to extract a " + channel + " image from a gray level image!")
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
    if self.dtype is not None:
      image = image.astype(self.dtype)
    return image


  def __call__(self, image, annotations = None):
    """Just perform gray scale conversion, ignore the annotations."""
    assert isinstance(image, numpy.ndarray) and image.ndim in (2,3)
    # convert to grayscale
    image = self.color_channel(image)
    return self.data_type(image)
