#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from sklearn.base import TransformerMixin, BaseEstimator
from skimage.transform import resize
import numpy as np
from sklearn.utils import check_array


class Scale(TransformerMixin, BaseEstimator):

    """
    Simple scales an images

    Parameters
    -----------

      target_img_size: tuple
         Target image size


    """

    def __init__(self, target_img_size, **kwargs):

        self.target_img_size = target_img_size

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Resize an image given a shape

        Parameters
        ----------

          img:
             Input image

          target_img_size: tuple
             Target image size

        """

        def _resize(x):
            return resize(x, self.target_img_size, anti_aliasing=True)

        X = check_array(X, allow_nd=True)

        if X.ndim <= 3 and X.ndim >= 4:
            raise ValueError(f"Invalid image shape {X.shape}")

        if X.ndim == 3:
            # Checking if it's bob format CxHxW
            if X.shape[0] == 3:
                X = np.moveaxis(X, -1, 0)
            return _resize(X)

        # Batch of images
        if X.ndim == 4:
            # Checking if it's bob format NxCxHxW
            if X.shape[1] == 3:
                X = np.moveaxis(X, 1, -1)
            return [_resize(x) for x in X]
