# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""
Color functionalitites from old bob.ip.color
"""
import numpy as np


def rgb_to_gray(image):
    """
    Converts an RGB image to a grayscale image.
    The formula is:
    GRAY = 0.299 * R + 0.587 * G + 0.114 * B

    Parameters
    ----------

    image : numpy.ndarray
        An image in RGB format (channels first): For an ND array (N >= 3),


    """

    assert image.ndim == 3, "The image should have 3 dimensions"

    R = image[0, :, :]
    G = image[1, :, :]
    B = image[2, :, :]

    return 0.299 * R + 0.587 * G + 0.114 * B


def gray_to_rgb(image):
    """
    Converts a grayscale image to an RGB image.
    The formula is:
    R = G = B = GRAY

    Parameters
    ----------

    image : numpy.ndarray
        An image in grayscale format (channels first): For an ND array (N >= 3),


    """

    assert image.ndim == 2, "The image should have 2 dimensions"

    return np.stack((image, image, image), axis=0)
