# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""
Color functionalitites from old bob.ip.color
"""
import numpy as np
from bob.io.image import to_matplotlib, to_bob
from PIL import Image


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

    if isinstance(image, np.floating):

        return (
            to_bob(
                np.array(
                    Image.fromarray(
                        (to_matplotlib(image) * 255).astype("uint8")
                    ).convert("HSV")
                )
            )
            / 255.0
        )
    else:
        return to_bob(np.array(Image.fromarray((to_matplotlib(image))).convert("HSV")))


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
