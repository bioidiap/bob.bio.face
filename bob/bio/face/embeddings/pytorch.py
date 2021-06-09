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
        memory_demanding=False,
        **kwargs
    ):

        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.model = None
        self.preprocessor = preprocessor
        self.memory_demanding = memory_demanding

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

        def _transform(X):
            return self.model(X).detach().numpy()

        if self.memory_demanding:
            return np.array([_transform(x[None, ...]) for x in X])
        else:
            return _transform(X)

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

    def __init__(self, memory_demanding=False):

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

        super(AFFFE_2021, self).__init__(
            checkpoint_path, config, memory_demanding=memory_demanding
        )

    def _load_model(self):

        import torch

        MainModel = imp.load_source("MainModel", self.config)
        network = torch.load(self.checkpoint_path)
        network.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        network.to(device)

        self.model = network


def _get_iresnet_file():
    urls = [
        "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet-91a5de61.tar.gz",
        "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet-91a5de61.tar.gz",
    ]

    return get_file(
        "iresnet-91a5de61.tar.gz",
        urls,
        cache_subdir="data/pytorch/iresnet-91a5de61/",
        file_hash="3976c0a539811d888ef5b6217e5de425",
        extract=True,
    )


class IResnet34(PyTorchModel):
    """
    ArcFace model (RESNET 34) from Insightface ported to pytorch
    """

    def __init__(self, memory_demanding=False):

        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet-91a5de61.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet-91a5de61.tar.gz",
        ]

        filename = _get_iresnet_file()

        path = os.path.dirname(filename)
        config = os.path.join(path, "iresnet.py")
        checkpoint_path = os.path.join(path, "iresnet34-5b0d0e90.pth")

        super(IResnet34, self).__init__(
            checkpoint_path, config, memory_demanding=memory_demanding
        )

    def _load_model(self):

        model = imp.load_source("module", self.config).iresnet34(self.checkpoint_path)
        self.model = model


class IResnet50(PyTorchModel):
    """
    ArcFace model (RESNET 50) from Insightface ported to pytorch
    """

    def __init__(self, memory_demanding=False):

        filename = _get_iresnet_file()

        path = os.path.dirname(filename)
        config = os.path.join(path, "iresnet.py")
        checkpoint_path = os.path.join(path, "iresnet50-7f187506.pth")

        super(IResnet50, self).__init__(
            checkpoint_path, config, memory_demanding=memory_demanding
        )

    def _load_model(self):

        model = imp.load_source("module", self.config).iresnet50(self.checkpoint_path)
        self.model = model


class IResnet100(PyTorchModel):
    """
    ArcFace model (RESNET 100) from Insightface ported to pytorch
    """

    def __init__(self, memory_demanding=False):

        filename = _get_iresnet_file()

        path = os.path.dirname(filename)
        config = os.path.join(path, "iresnet.py")
        checkpoint_path = os.path.join(path, "iresnet100-73e07ba7.pth")

        super(IResnet100, self).__init__(
            checkpoint_path, config, memory_demanding=memory_demanding
        )

    def _load_model(self):

        model = imp.load_source("module", self.config).iresnet100(self.checkpoint_path)
        self.model = model


def iresnet_template(embedding, annotation_type, fixed_positions=None):
    # DEFINE CROPPING
    cropped_image_size = (112, 112)
    if annotation_type == "eyes-center":
        # Hard coding eye positions for backward consistency
        cropped_positions = {
            "leye": (55, 81),
            "reye": (55, 42),
        }
    else:
        cropped_positions = dnn_default_cropping(cropped_image_size, annotation_type)

    transformer = embedding_transformer(
        cropped_image_size=cropped_image_size,
        embedding=embedding,
        cropped_positions=cropped_positions,
        fixed_positions=fixed_positions,
        color_channel="rgb",
        annotator="mtcnn",
    )

    algorithm = Distance()

    return VanillaBiometricsPipeline(transformer, algorithm)


def iresnet34(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the Resnet34 pipeline which will crop the face :math:`112 \times 112` and 
    use the :py:class:`IResnet34` to extract the features


    code referenced from https://raw.githubusercontent.com/nizhib/pytorch-insightface/master/insightface/iresnet.py
    https://github.com/nizhib/pytorch-insightface


    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return iresnet_template(
        embedding=IResnet34(memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def iresnet50(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the Resnet50 pipeline which will crop the face :math:`112 \times 112` and 
    use the :py:class:`IResnet50` to extract the features


    code referenced from https://raw.githubusercontent.com/nizhib/pytorch-insightface/master/insightface/iresnet.py
    https://github.com/nizhib/pytorch-insightface


    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return iresnet_template(
        embedding=IResnet50(memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def iresnet100(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the Resnet100 pipeline which will crop the face :math:`112 \times 112` and 
    use the :py:class:`IResnet100` to extract the features


    code referenced from https://raw.githubusercontent.com/nizhib/pytorch-insightface/master/insightface/iresnet.py
    https://github.com/nizhib/pytorch-insightface


    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return iresnet_template(
        embedding=IResnet100(memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def afffe_baseline(annotation_type, fixed_positions=None, memory_demanding=False):
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
        embedding=AFFFE_2021(memory_demanding=memory_demanding),
        cropped_positions=cropped_positions,
        fixed_positions=fixed_positions,
        color_channel="rgb",
        annotator="mtcnn",
    )

    algorithm = Distance()

    return VanillaBiometricsPipeline(transformer, algorithm)
