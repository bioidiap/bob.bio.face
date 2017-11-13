#!/usr/bin/env python
# encoding: utf-8

"""Features for face recognition"""

import numpy
import bob.io.base
from bob.bio.base.extractor import Extractor

import bob.io.base
import tensorflow as tf
import bob.ip.base

from bob.learn.drgan.networks import DRGAN

class LuanExtractor(Extractor):
  """

  **Parameters:**

  """

  def __init__(self):

    Extractor.__init__(self, skip_extractor_training=True)
    
    # not relevant (discriminator)
    self.identity_dim = 208 
    self.conditional_dim = 13
    
    # relevant and to be checked with the saved model
    self.latent_dim = 320
    self.image_size = [96, 96, 3]

    import tensorflow as tf
    self.session = tf.Session()

    self.drgan = DRGAN(image_size=96, z_dim=100, gf_dim=32, gfc_dim=320, c_dim=3, checkpoint_dir='')
    data_shape = [1] + self.image_size
    
    self.X = tf.placeholder(tf.float32, shape=data_shape)
    self.encode = self.drgan.generator_encoder(self.X, is_reuse=False, is_training=False)
    
    self.saver = tf.train.Saver()
   

  def __call__(self, image):
    """__call__(image) -> feature

    Extract features

    **Parameters:**

    image : 3D :py:class:`numpy.ndarray` (floats)
      The image to extract the features from.

    **Returns:**

    feature : 2D :py:class:`numpy.ndarray` (floats)
      The extracted features
    """
    def bob2skimage(bob_image):
      """
      Convert bob color image to the skcit image
      """

      if len(bob_image.shape) == 3:
        skimage = numpy.zeros(shape=(bob_image.shape[1], bob_image.shape[2], 3), dtype=numpy.uint8)
        skimage[:,:,0] = bob_image[0, :,:]
        skimage[:,:,1] = bob_image[1, :,:]
        skimage[:,:,2] = bob_image[2, :,:]
      else:
        skimage = numpy.zeros(shape=(bob_image.shape[0], bob_image.shape[1], 1))
        skimage[:,:,0] = bob_image[:,:]
      return skimage

    def rescaleToUint8(image):
      result = numpy.zeros_like(image)
      for channel in range(image.shape[2]):
        min_image = numpy.min(image[:, :, channel])
        max_image = numpy.max(image[:, :, channel])
        if (max_image - min_image) != 0:
          result[:, :, channel] = 255.0*((image[:, :, channel] - min_image) / (max_image - min_image))
        else:
          result[:, :, channel] = 0 
        result = result.astype('uint8')
      return result
    
    # encode the provided image
    image = rescaleToUint8(image)
    #from matplotlib import pyplot
    #pyplot.imshow(numpy.rollaxis(image, 0, 3))
    #pyplot.show()
    #
    image = bob2skimage(image)
    image = numpy.array(image/127.5 - 1).astype(numpy.float32)
    
    shape = [1] + list(image.shape)
    img = numpy.reshape(image, tuple(shape))
    #
    #pyplot.imshow(image)
    #pyplot.show()


    encoded_id = self.session.run(self.encode, feed_dict={self.X : img})
    return encoded_id

  # re-define the train function to get it non-documented
  def train(*args, **kwargs): raise NotImplementedError("This function is not implemented and should not be called.")

  def load(self, extractor_file):
    self.saver.restore(self.session, extractor_file)
