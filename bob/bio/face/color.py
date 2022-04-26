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


def rgb_to_hsv(image):
    """
    Converts an RGB image to an HSV image.

    Parameters
    ----------

    image : numpy.ndarray
        An image in RGB format (channels first): For an ND array (N >= 3),


    """

    assert image.ndim == 3, "The image should have 3 dimensions"

    image_hsv = np.zeros(image.shape)

    R = image[0, :, :]
    G = image[1, :, :]
    B = image[2, :, :]

    max_value = np.max(image, axis=0)
    min_value = np.min(image, axis=0)
    delta = max_value - min_value

    # Hue
    image_hsv[0, :, :] = np.arctan2(B - G, R - G)
    image_hsv[0, :, :] = (image_hsv[0, :, :] + np.pi) % (2 * np.pi)

    # Saturation
    image_hsv[1, :, :] = np.maximum(max_value, 0.00001)
    image_hsv[1, :, :] /= np.maximum(delta, 0.00001)
    image_hsv[1, :, :] = np.minimum(image_hsv[1, :, :], 1.0)

    # Value
    image_hsv[2, :, :] = max_value

    return image_hsv


def rgb_to_yuv(image):
    """
    Converts an RGB image to an YUV image.

    Parameters
    ----------

    image : numpy.ndarray
        An image in RGB format (channels first): For an ND array (N >= 3),


    """

    assert image.ndim == 3, "The image should have 3 dimensions"

    image_yuv = np.zeros(image.shape)

    R = image[0, :, :]
    G = image[1, :, :]
    B = image[2, :, :]

    # Y
    image_yuv[0, :, :] = 0.299 * R + 0.587 * G + 0.114 * B

    # U
    image_yuv[1, :, :] = 0.492 * (B - image_yuv[0, :, :])

    # V
    image_yuv[2, :, :] = 0.877 * (R - image_yuv[0, :, :])

    return image_yuv
