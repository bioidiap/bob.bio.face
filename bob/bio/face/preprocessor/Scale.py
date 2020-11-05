from sklearn.preprocessing import FunctionTransformer
from skimage.transform import resize
from sklearn.utils import check_array
from bob.io.image import to_matplotlib, to_bob


def scale(images, target_img_size):
    """Scales a list of images to the target size

    Parameters
    ----------
    images : array_like
        A list of images (in Bob format) to be scaled to the target size
    target_img_size : int or tuple
        A tuple of size 2 as (H, W) or an integer where H==W

    Returns
    -------
    numpy.ndarray
        Scaled images
    """
    if isinstance(target_img_size, int):
        target_img_size = (target_img_size, target_img_size)

    images = check_array(images, allow_nd=True)
    images = to_matplotlib(images)

    # images are always batched
    output_shape = tuple(target_img_size)
    output_shape = tuple(images.shape[0:1]) + output_shape
    images = resize(images, output_shape=output_shape)

    return to_bob(images)


def Scale(target_img_size):
    """
    A transformer that scales images.
    It accepts a list of inputs

    Parameters
    -----------

      target_img_size: tuple
         Target image size, specified as a tuple of (H, W)


    """
    return FunctionTransformer(
        func=scale, validate=False, kw_args=dict(target_img_size=target_img_size)
    )