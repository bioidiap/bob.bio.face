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
from bob.bio.face.utils import (
    dnn_default_cropping,
    embedding_transformer,
)

from bob.bio.base.pipelines.vanilla_biometrics import (
    Distance,
    VanillaBiometricsPipeline,
)


class PyTorchModel(TransformerMixin, BaseEstimator):
    """
    Base Transformer using pytorch models


    Parameters
    ----------

    checkpoint_path: str
       Path containing the checkpoint

    config:
        Path containing some configuration file (e.g. .json, .prototxt)

    preprocessor:
        A function that will transform the data right before forward. The default transformation is `X/255`

    """

    def __init__(
        self,
        checkpoint_path=None,
        config=None,
        preprocessor=lambda x: x / 255,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.model = None
        self.preprocessor = preprocessor

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
        X = self.preprocessor(X)

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


def afffe_baseline(annotation_type, fixed_positions=None):
    """
    Get the AFFFE pipeline which will crop the face :math:`224 \times 224`
    use the :py:class:`AFFFE_2021`

    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image
    """

    # DEFINE CROPPING
    cropped_image_size = (224, 224)

    if annotation_type == "eyes-center":
        # Hard coding eye positions for backward consistency
        cropped_positions = {"leye": (110, 144), "reye": (110, 96)}
    else:
        cropped_positions = dnn_default_cropping(cropped_image_size, annotation_type)

    transformer = embedding_transformer(
        cropped_image_size=cropped_image_size,
        embedding=AFFFE_2021(),
        cropped_positions=cropped_positions,
        fixed_positions=fixed_positions,
        color_channel="rgb",
        annotator="mtcnn",
    )

    algorithm = Distance()

    return VanillaBiometricsPipeline(transformer, algorithm)
