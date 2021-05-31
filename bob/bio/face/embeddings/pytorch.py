#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Yu Linghu & Xinyi Zhang <yu.linghu@uzh.ch, xinyi.zhang@uzh.ch>
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
import numpy as np
import imp
import os
from bob.extension.download import get_file


class PyTorchModel(TransformerMixin, BaseEstimator):
    """
    """

    def __init__(self, checkpoint_path=None, config=None, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.model = None

    def transform(self, X):
        """__call__(image) -> feature

        Extracts the features from the given image.

        **Parameters:**

        image : 2D :py:class:`numpy.ndarray` (floats)
        The image to extract the features from.

        **Returns:**

        feature : 2D or 3D :py:class:`numpy.ndarray` (floats)
        The list of features extracted from the image.
        """
        import torch

        if self.model is None:
            self._load_model()
        X = check_array(X, allow_nd=True)
        X = torch.Tensor(X)
        X = X / 255

        return self.model(X).detach().numpy()

    def __getstate__(self):
        # Handling unpicklable objects

        d = self.__dict__.copy()
        d["model"] = None
        return d

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}


class AFFFE_2021(PyTorchModel):
    """
    AFFFE from https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/AFFFE-42a53f19.tar.gz

    """

    def __init__(self):

        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/AFFFE-42a53f19.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/AFFFE-42a53f19.tar.gz",
        ]

        filename = get_file(
            "AFFFE-42a53f19.tar.gz",
            urls,
            cache_subdir="data/pytorch/AFFFE-42a53f19.tar.gz",
            file_hash="1358bbcda62cb59b85b2418ef1f81e9b",
            extract=True,
        )
        path = os.path.dirname(filename)
        config = os.path.join(path, "AFFFE.py")
        checkpoint_path = os.path.join(path, "AFFFE.pth")

        super(AFFFE_2021, self).__init__(checkpoint_path, config)

    def _load_model(self):

        import torch

        MainModel = imp.load_source("MainModel", self.config)
        network = torch.load(self.checkpoint_path)
        network.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        network.to(device)

        self.model = network

