#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Yu Linghu & Xinyi Zhang <yu.linghu@uzh.ch, xinyi.zhang@uzh.ch>
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import imp
import os

import numpy as np
import torch

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

from bob.bio.base.algorithm import Distance
from bob.bio.base.pipelines import PipelineSimple
from bob.bio.face.annotator import MTCNN
from bob.bio.face.pytorch.facexzoo import FaceXZooModelFactory
from bob.bio.face.utils import (
    cropped_positions_arcface,
    dnn_default_cropping,
    embedding_transformer,
)
from bob.extension.download import get_file


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
        device=None,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.model = None
        self.preprocessor = preprocessor
        self.memory_demanding = memory_demanding
        self.device = torch.device(
            device or "cuda" if torch.cuda.is_available() else "cpu"
        )

    def transform(self, X):
        """__call__(image) -> feature

        Extracts the features from the given image.

        Parameters
        ----------

        image : 2D :py:class:`numpy.ndarray` (floats)
        The image to extract the features from.

        Returns
        -------

        feature : 2D or 3D :py:class:`numpy.ndarray` (floats)
        The list of features extracted from the image.
        """
        import torch

        if self.model is None:
            self._load_model()
        X = check_array(X, allow_nd=True)
        X = torch.Tensor(X)
        with torch.no_grad():
            X = self.preprocessor(X)

        def _transform(X):
            with torch.no_grad():
                return self.model(X.to(self.device)).cpu().detach().numpy()

        if self.memory_demanding:
            features = np.array([_transform(x[None, ...]) for x in X])

            # If we ndim is > than 3. We should stack them all
            # The enroll_features can come from a source where there are `N` samples containing
            # nxd samples
            if features.ndim >= 3:
                features = np.vstack(features)

            return features

        else:
            return _transform(X)

    def __getstate__(self):
        # Handling unpicklable objects

        d = self.__dict__.copy()
        d["model"] = None
        return d

    def _more_tags(self):
        return {"requires_fit": False}

    def place_model_on_device(self):
        if self.model is not None:
            self.model.to(self.device)


class RunnableModel(PyTorchModel):
    """
    Runnable pytorch model

    With this class it is possible to pass a loaded pytorch model as an argument.

    The parameter `model` can be set in two ways.
    The first one via an pytorch module instance. Like in the example below:

       >>> model = my_py_torch_model.load("PATH")
       >>> transformer = RunnableModel(model) #doctest: +SKIP


    The second one is via a `callable` function.
    The mode is useful while running under Dask.
    This avoids the serialization and data transfer of the model.
    At the Idiap's network, this can be a problem.
    See the example below on how to set the `model` as a callable:

       >>> from functools import partial
       >>> model = partial(my_py_torch_model.load,"PATH")
       >>> transformer = RunnableModel(model) #doctest: +SKIP



    Parameters:
    ----------
        model:
          Loaded pytorch model OR a function that loads the model.
          Providing a function that loads the model might be useful for the
          idiap grid

        preprocessor:
          A function that will transform the data right before forward. The default transformation is `X/255`

        memory_demanding:

        device:

    """

    def __init__(
        self,
        model,
        preprocessor=lambda x: x / 255,
        memory_demanding=False,
        device=None,
        **kwargs,
    ):
        super(RunnableModel, self).__init__(
            preprocessor=preprocessor,
            memory_demanding=memory_demanding,
            device=device,
            **kwargs,
        )

        if callable(model):
            self.model = None
            self._model_fn = model
            self.is_loaded_by_function = True
        else:
            self.model = model
            self.model.eval()
            self.is_loaded_by_function = False

    def _load_model(self):
        self.model = self._model_fn()
        self.model.eval()

    def __getstate__(self):
        if self.is_loaded_by_function:
            return super(RunnableModel, self).__getstate__()


class AFFFE_2021(PyTorchModel):
    """
    AFFFE Pytorch network that extracts 1000-dimensional features, trained by Manuel Gunther, as described in [LGB18]_

    """

    def __init__(self, memory_demanding=False, device=None, **kwargs):

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
            checkpoint_path,
            config,
            memory_demanding=memory_demanding,
            device=device,
            **kwargs,
        )

    def _load_model(self):
        import torch

        _ = imp.load_source("MainModel", self.config)
        self.model = torch.load(self.checkpoint_path, map_location=self.device)

        self.model.eval()
        self.place_model_on_device()


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

    def __init__(
        self,
        preprocessor=lambda x: (x - 127.5) / 128.0,
        memory_demanding=False,
        device=None,
        **kwargs,
    ):

        filename = _get_iresnet_file()

        path = os.path.dirname(filename)
        config = os.path.join(path, "iresnet.py")
        checkpoint_path = os.path.join(path, "iresnet34-5b0d0e90.pth")

        super(IResnet34, self).__init__(
            checkpoint_path,
            config,
            memory_demanding=memory_demanding,
            preprocessor=preprocessor,
            device=device,
            **kwargs,
        )

    def _load_model(self):

        model = imp.load_source("module", self.config).iresnet34(
            self.checkpoint_path
        )
        self.model = model

        self.model.eval()
        self.place_model_on_device()


class IResnet50(PyTorchModel):
    """
    ArcFace model (RESNET 50) from Insightface ported to pytorch
    """

    def __init__(
        self,
        preprocessor=lambda x: (x - 127.5) / 128.0,
        memory_demanding=False,
        device=None,
        **kwargs,
    ):

        filename = _get_iresnet_file()

        path = os.path.dirname(filename)
        config = os.path.join(path, "iresnet.py")
        checkpoint_path = os.path.join(path, "iresnet50-7f187506.pth")

        super(IResnet50, self).__init__(
            checkpoint_path,
            config,
            memory_demanding=memory_demanding,
            preprocessor=preprocessor,
            device=device,
            **kwargs,
        )

    def _load_model(self):

        model = imp.load_source("module", self.config).iresnet50(
            self.checkpoint_path
        )
        self.model = model

        self.model.eval()
        self.place_model_on_device()


class IResnet100(PyTorchModel):
    """
    ArcFace model (RESNET 100) from Insightface ported to pytorch
    """

    def __init__(
        self,
        preprocessor=lambda x: (x - 127.5) / 128.0,
        memory_demanding=False,
        device=None,
        **kwargs,
    ):

        filename = _get_iresnet_file()

        path = os.path.dirname(filename)
        config = os.path.join(path, "iresnet.py")
        checkpoint_path = os.path.join(path, "iresnet100-73e07ba7.pth")

        super(IResnet100, self).__init__(
            checkpoint_path,
            config,
            memory_demanding=memory_demanding,
            preprocessor=preprocessor,
            device=device,
            **kwargs,
        )

    def _load_model(self):
        model = imp.load_source("module", self.config).iresnet100(
            self.checkpoint_path
        )
        self.model = model

        self.model.eval()
        self.place_model_on_device()


class OxfordVGG2Resnets(PyTorchModel):
    """
    Get the transformer for the resnet based models from Oxford.
    All these models were training the the VGG2 dataset.

    Models taken from: https://www.robots.ox.ac.uk/~albanie


    Parameters
    ----------

    model_name: str
      One of the 4 models available (`resnet50_scratch_dag`, `resnet50_ft_dag`, `senet50_ft_dag`, `senet50_scratch_dag`).

    """

    def __init__(
        self,
        model_name,
        memory_demanding=False,
        device=None,
        **kwargs,
    ):

        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/oxford_resnet50_vgg2.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/oxford_resnet50_vgg2.tar.gz",
        ]

        filename = get_file(
            "oxford_resnet50_vgg2.tar.gz",
            urls,
            cache_subdir="data/pytorch/oxford_resnet50_vgg2/",
            file_hash="c8e1ed3715d83647b4a02e455213aaf0",
            extract=True,
        )

        models_available = [
            "resnet50_scratch_dag",
            "resnet50_ft_dag",
            "senet50_ft_dag",
            "senet50_scratch_dag",
        ]
        if model_name not in models_available:
            raise ValueError(
                f"Invalid model {model_name}. The models available are {models_available}"
            )

        self.model_name = model_name
        path = os.path.dirname(filename)
        config = os.path.join(path, model_name, f"{model_name}.py")
        checkpoint_path = os.path.join(path, model_name, f"{model_name}.pth")

        super(OxfordVGG2Resnets, self).__init__(
            checkpoint_path,
            config,
            memory_demanding=memory_demanding,
            preprocessor=self.dag_preprocessor,
            device=device,
            **kwargs,
        )

    def dag_preprocessor(self, X):
        """
        Normalize using `self.meta`

        Caffe has the shape `H x W x C` and the chanel is BGR and

        """

        # Convert to H x W x C
        # X = torch.moveaxis(X, 1, 3)

        # Subtracting
        X[:, 0, :, :] = (X[:, 0, :, :] - self.meta["mean"][0]) / self.meta[
            "std"
        ][0]
        X[:, 1, :, :] = (X[:, 1, :, :] - self.meta["mean"][1]) / self.meta[
            "std"
        ][1]
        X[:, 2, :, :] = (X[:, 2, :, :] - self.meta["mean"][2]) / self.meta[
            "std"
        ][2]

        return X

    def _load_model(self):
        if self.model_name == "resnet50_scratch_dag":
            model = imp.load_source("module", self.config).resnet50_scratch_dag(
                weights_path=self.checkpoint_path
            )
        elif self.model_name == "resnet50_ft_dag":
            model = imp.load_source("module", self.config).resnet50_ft_dag(
                weights_path=self.checkpoint_path
            )
        elif self.model_name == "senet50_scratch_dag":
            model = imp.load_source("module", self.config).senet50_scratch_dag(
                weights_path=self.checkpoint_path
            )
        else:
            model = imp.load_source("module", self.config).senet50_ft_dag(
                weights_path=self.checkpoint_path
            )

        self.model = model
        self.meta = self.model.meta

        self.model.eval()
        self.place_model_on_device()

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
        with torch.no_grad():
            X = self.preprocessor(X)

        def _transform(X):
            with torch.no_grad():
                # Fetching the pool5_7x7_s1 layer which
                return (
                    self.model(X.to(self.device))[1]
                    .cpu()
                    .detach()
                    .numpy()[:, :, 0, 0]
                )

        if self.memory_demanding:
            features = np.array([_transform(x[None, ...]) for x in X])

            # If we ndim is > than 3. We should stack them all
            # The enroll_features can come from a source where there are `N` samples containing
            # nxd samples
            if features.ndim >= 3:
                features = np.vstack(features)

            return features

        else:
            return _transform(X)


class IResnet100Elastic(PyTorchModel):
    """
    iResnet100 model from the paper.

    Boutros, Fadi, et al. "ElasticFace: Elastic Margin Loss for Deep Face Recognition." arXiv preprint arXiv:2109.09416 (2021).
    """

    def __init__(
        self,
        preprocessor=lambda x: (x - 127.5) / 128.0,
        memory_demanding=False,
        device=None,
        **kwargs,
    ):

        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet100-elastic.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet100-elastic.tar.gz",
        ]

        filename = get_file(
            "iresnet100-elastic.tar.gz",
            urls,
            cache_subdir="data/pytorch/iresnet100-elastic/",
            file_hash="0ac36db3f0f94930993afdb27faa4f02",
            extract=True,
        )

        path = os.path.dirname(filename)
        config = os.path.join(path, "iresnet.py")
        checkpoint_path = os.path.join(path, "iresnet100-elastic.pt")

        super(IResnet100Elastic, self).__init__(
            checkpoint_path,
            config,
            memory_demanding=memory_demanding,
            preprocessor=preprocessor,
            device=device,
            **kwargs,
        )

    def _load_model(self):
        model = imp.load_source("module", self.config).iresnet100(
            self.checkpoint_path
        )
        self.model = model

        self.model.eval()
        self.place_model_on_device()


class FaceXZooModel(PyTorchModel):
    """
    FaceXZoo models
    """

    def __init__(
        self,
        preprocessor=lambda x: (x - 127.5) / 128.0,
        memory_demanding=False,
        device=None,
        arch="AttentionNet",
        **kwargs,
    ):

        self.arch = arch
        _model = FaceXZooModelFactory(self.arch)
        filename = _model.get_facexzoo_file()
        config = None
        path = os.path.dirname(filename)
        checkpoint_path = os.path.join(path, self.arch + ".pt")

        super(FaceXZooModel, self).__init__(
            checkpoint_path,
            config,
            memory_demanding=memory_demanding,
            preprocessor=preprocessor,
            device=device,
            **kwargs,
        )

    def _load_model(self):

        _model = FaceXZooModelFactory(self.arch)
        self.model = _model.get_model()

        model_dict = self.model.state_dict()

        pretrained_dict = torch.load(
            self.checkpoint_path, map_location=torch.device("cpu")
        )["state_dict"]

        new_pretrained_dict = {}
        for k in model_dict:
            new_pretrained_dict[k] = pretrained_dict["backbone." + k]
        model_dict.update(new_pretrained_dict)
        self.model.load_state_dict(model_dict)

        self.model.eval()
        self.place_model_on_device()


def iresnet_template(embedding, annotation_type, fixed_positions=None):
    # DEFINE CROPPING
    cropped_image_size = (112, 112)

    if (
        annotation_type == "eyes-center"
        or annotation_type == "bounding-box"
        or isinstance(annotation_type, list)
    ):
        # Hard coding eye positions for backward consistency
        # cropped_positions = {
        cropped_positions = cropped_positions_arcface(annotation_type)
        if annotation_type == "bounding-box":
            # This will allow us to use `BoundingBoxAnnotatorCrop`
            cropped_positions.update(
                {"topleft": (0, 0), "bottomright": cropped_image_size}
            )

    else:
        cropped_positions = dnn_default_cropping(
            cropped_image_size, annotation_type
        )

    annotator = MTCNN(min_size=40, factor=0.709, thresholds=(0.1, 0.2, 0.2))
    transformer = embedding_transformer(
        cropped_image_size=cropped_image_size,
        embedding=embedding,
        cropped_positions=cropped_positions,
        fixed_positions=fixed_positions,
        color_channel="rgb",
        annotator=annotator,
    )

    algorithm = Distance()

    return PipelineSimple(transformer, algorithm)


def AttentionNet(
    annotation_type,
    fixed_positions=None,
    memory_demanding=False,
    device=torch.device("cpu"),
):
    """
    Get the AttentionNet pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`AttentionNet` to extract the features


    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `pipeline simple` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """
    return iresnet_template(
        embedding=FaceXZooModel(
            arch="AttentionNet",
            memory_demanding=memory_demanding,
            device=device,
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def ResNeSt(
    annotation_type,
    fixed_positions=None,
    memory_demanding=False,
    device=torch.device("cpu"),
):
    """
    Get the ResNeSt pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`ResNeSt` to extract the features

    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `pipeline simple` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return iresnet_template(
        embedding=FaceXZooModel(
            arch="ResNeSt", memory_demanding=memory_demanding, device=device
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def MobileFaceNet(
    annotation_type,
    fixed_positions=None,
    memory_demanding=False,
    device=torch.device("cpu"),
):
    """
    Get the MobileFaceNet pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`MobileFaceNet` to extract the features

    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `pipeline simple` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return iresnet_template(
        embedding=FaceXZooModel(
            arch="MobileFaceNet",
            memory_demanding=memory_demanding,
            device=device,
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def ResNet(
    annotation_type,
    fixed_positions=None,
    memory_demanding=False,
    device=torch.device("cpu"),
):
    """
    Get the ResNet pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`ResNet` to extract the features


    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `pipeline simple` pipeline.


    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return iresnet_template(
        embedding=FaceXZooModel(
            arch="ResNet", memory_demanding=memory_demanding, device=device
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def EfficientNet(
    annotation_type,
    fixed_positions=None,
    memory_demanding=False,
    device=torch.device("cpu"),
):
    """
    Get the EfficientNet pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`EfficientNet` to extract the features


    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `pipeline simple` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return iresnet_template(
        embedding=FaceXZooModel(
            arch="EfficientNet",
            memory_demanding=memory_demanding,
            device=device,
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def TF_NAS(
    annotation_type,
    fixed_positions=None,
    memory_demanding=False,
    device=torch.device("cpu"),
):
    """
    Get the TF_NAS pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`TF-NAS` to extract the features


    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `pipeline simple` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return iresnet_template(
        embedding=FaceXZooModel(
            arch="TF-NAS", memory_demanding=memory_demanding, device=device
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def HRNet(
    annotation_type,
    fixed_positions=None,
    memory_demanding=False,
    device=torch.device("cpu"),
):
    """
    Get the HRNet pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`HRNet` to extract the features


    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `pipeline simple` pipeline.


    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return iresnet_template(
        embedding=FaceXZooModel(
            arch="HRNet", memory_demanding=memory_demanding, device=device
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def ReXNet(
    annotation_type,
    fixed_positions=None,
    memory_demanding=False,
    device=torch.device("cpu"),
):
    """
    Get the ReXNet pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`ReXNet` to extract the features

    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `pipeline simple` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return iresnet_template(
        embedding=FaceXZooModel(
            arch="ReXNet", memory_demanding=memory_demanding
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def GhostNet(
    annotation_type,
    fixed_positions=None,
    memory_demanding=False,
    device=torch.device("cpu"),
):
    """
    Get the GhostNet pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`GhostNet` to extract the features


    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `pipeline simple` pipeline.


    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return iresnet_template(
        embedding=FaceXZooModel(
            arch="GhostNet", memory_demanding=memory_demanding, device=device
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def iresnet34(
    annotation_type,
    fixed_positions=None,
    memory_demanding=False,
    device=torch.device("cpu"),
):
    """
    Get the Resnet34 pipeline which will crop the face :math:`112 \\times 112` and
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
        embedding=IResnet34(memory_demanding=memory_demanding, device=device),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def iresnet50(
    annotation_type,
    fixed_positions=None,
    memory_demanding=False,
    device=torch.device("cpu"),
):
    """
    Get the Resnet50 pipeline which will crop the face :math:`112 \\times 112` and
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
        embedding=IResnet50(memory_demanding=memory_demanding, device=device),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def iresnet100(
    annotation_type,
    fixed_positions=None,
    memory_demanding=False,
    device=torch.device("cpu"),
):
    """
    Get the Resnet100 pipeline which will crop the face :math:`112 \\times 112` and
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
        embedding=IResnet100(memory_demanding=memory_demanding, device=device),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def iresnet100_elastic(
    annotation_type,
    fixed_positions=None,
    memory_demanding=False,
    device=torch.device("cpu"),
):
    """
    Get the Resnet100 pipeline which will crop the face :math:`112 \\times 112` and
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
        embedding=IResnet100Elastic(
            memory_demanding=memory_demanding, device=device
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def afffe_baseline(
    annotation_type,
    fixed_positions=None,
    memory_demanding=False,
    device=torch.device("cpu"),
):
    """
    Get the AFFFE pipeline which will crop the face :math:`224 \\times 224`
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
        cropped_positions = dnn_default_cropping(
            cropped_image_size, annotation_type
        )

    transformer = embedding_transformer(
        cropped_image_size=cropped_image_size,
        embedding=AFFFE_2021(memory_demanding=memory_demanding, device=device),
        cropped_positions=cropped_positions,
        fixed_positions=fixed_positions,
        color_channel="rgb",
        annotator="mtcnn",
    )

    algorithm = Distance()
    from bob.bio.base.pipelines import PipelineSimple

    return PipelineSimple(transformer, algorithm)


def oxford_vgg2_resnets(
    model_name, annotation_type, fixed_positions=None, memory_demanding=False
):
    """
    Get the pipeline for the resnet based models from Oxford.
    All these models were training the the VGG2 dataset.

    Models taken from: https://www.robots.ox.ac.uk/~albanie

    Parameters
    ----------
      model_name: str
         One of the 4 models available (`resnet50_scratch_dag`, `resnet50_ft_dag`, `senet50_ft_dag`, `senet50_scratch_dag`).

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image
    """

    # DEFINE CROPPING
    cropped_image_size = (224, 224)

    if annotation_type == "eyes-center":
        # Coordinates taken from : https://www.merlin.uzh.ch/contributionDocument/download/14240
        cropped_positions = {"leye": (100, 159), "reye": (100, 65)}
    else:
        cropped_positions = dnn_default_cropping(
            cropped_image_size, annotation_type
        )

    transformer = embedding_transformer(
        cropped_image_size=cropped_image_size,
        embedding=OxfordVGG2Resnets(
            model_name=model_name, memory_demanding=memory_demanding
        ),
        cropped_positions=cropped_positions,
        fixed_positions=fixed_positions,
        color_channel="rgb",
        annotator="mtcnn",
    )

    algorithm = Distance()
    from bob.bio.base.pipelines import PipelineSimple

    return PipelineSimple(transformer, algorithm)
