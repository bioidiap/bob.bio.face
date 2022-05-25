#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

# Tranformers based on tensorflow


import os

import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

from bob.bio.base.algorithm import Distance
from bob.bio.base.pipelines import PipelineSimple
from bob.bio.face.annotator import MTCNN
from bob.bio.face.utils import (
    cropped_positions_arcface,
    dnn_default_cropping,
    embedding_transformer,
)
from bob.extension.download import get_file


def to_channels_last(image):
    """Converts the image to channel_last format. This is the same format as in
    matplotlib, skimage, and etc.

    Parameters
    ----------
    image : `tf.Tensor`
        At least a 3 dimensional image. If the dimension is more than 3, the
        last 3 dimensions are assumed to be [C, H, W].

    Returns
    -------
    image : `tf.Tensor`
        The image in [..., H, W, C] format.

    Raises
    ------
    ValueError
        If dim of image is less than 3.
    """
    ndim = len(image.shape)
    if ndim < 3:
        raise ValueError(
            "The image needs to be at least 3 dimensional but it "
            "was {}".format(ndim)
        )
    axis_order = [1, 2, 0]
    shift = ndim - 3
    axis_order = list(range(ndim - 3)) + [n + shift for n in axis_order]
    return tf.transpose(a=image, perm=axis_order)


def to_channels_first(image):
    """Converts the image to channel_first format. This is the same format as
    in Bob's image and video.

    Parameters
    ----------
    image : `tf.Tensor`
        At least a 3 dimensional image. If the dimension is more than 3, the
        last 3 dimensions are assumed to be [H, W, C].

    Returns
    -------
    image : `tf.Tensor`
        The image in [..., C, H, W] format.

    Raises
    ------
    ValueError
        If dim of image is less than 3.
    """
    ndim = len(image.shape)
    if ndim < 3:
        raise ValueError(
            "The image needs to be at least 3 dimensional but it "
            "was {}".format(ndim)
        )
    axis_order = [2, 0, 1]
    shift = ndim - 3
    axis_order = list(range(ndim - 3)) + [n + shift for n in axis_order]
    return tf.transpose(a=image, perm=axis_order)


def sanderberg_rescaling():
    # FIXED_STANDARDIZATION from https://github.com/davidsandberg/facenet
    # [-0.99609375, 0.99609375]
    preprocessor = tf.keras.layers.experimental.preprocessing.Rescaling(
        scale=1 / 128, offset=-127.5 / 128
    )
    return preprocessor


class TensorflowTransformer(TransformerMixin, BaseEstimator):
    """
    Base Transformer for Tensorflow architectures.

    Parameters
    ----------

    checkpoint_path: str
       Path containing the checkpoint

    preprocessor:
        A function that will transform the data right before forward

    memory_demanding bool
        If `True`, the `transform` method will run one sample at the time.
        This is useful when there is not enough memory available to forward big chucks of data.
    """

    def __init__(
        self,
        checkpoint_path,
        preprocessor=None,
        memory_demanding=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.preprocessor = preprocessor
        self.memory_demanding = memory_demanding

    def load_model(self):
        self.model = tf.keras.models.load_model(
            self.checkpoint_path, compile=False
        )

    def transform(self, X):
        def _transform(X):
            X = tf.convert_to_tensor(X)
            X = to_channels_last(X)

            if X.shape[-3:] != self.model.input_shape[-3:]:
                raise ValueError(
                    f"Image shape {X.shape} not supported. Expected {self.model.input_shape}"
                )

            return self.inference(X).numpy()

        if self.model is None:
            self.load_model()

        X = check_array(X, allow_nd=True)

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

    def inference(self, X):
        if self.preprocessor is not None:
            X = self.preprocessor(tf.cast(X, "float32"))

        prelogits = self.model.predict_on_batch(X)
        embeddings = tf.math.l2_normalize(prelogits, axis=-1)
        return embeddings

    def _more_tags(self):
        return {"requires_fit": False}

    def __del__(self):
        self.model = None


class InceptionResnetv2_MsCeleb_CenterLoss_2018(TensorflowTransformer):
    """
    InceptionResnet v2 model trained in 2018 using the MSCeleb dataset in the context of the work:

    Freitas Pereira, Tiago, André Anjos, and Sébastien Marcel. "Heterogeneous face recognition using domain specific units." IEEE Transactions on Information Forensics and Security 14.7 (2018): 1803-1816.

    """

    def __init__(self, memory_demanding=False, **kwargs):

        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/inceptionresnetv2_msceleb_centerloss_2018.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/inceptionresnetv2_msceleb_centerloss_2018.tar.gz",
        ]

        filename = get_file(
            "inceptionresnetv2_msceleb_centerloss_2018.tar.gz",
            urls,
            cache_subdir="data/tensorflow/inceptionresnetv2_msceleb_centerloss_2018",
            file_hash="7c0aa46bba16c01768a38594a3b4c14d",
            extract=True,
        )
        checkpoint_path = os.path.dirname(filename)

        super(InceptionResnetv2_MsCeleb_CenterLoss_2018, self).__init__(
            checkpoint_path,
            preprocessor=tf.image.per_image_standardization,
            memory_demanding=memory_demanding,
            **kwargs,
        )


class InceptionResnetv2_Casia_CenterLoss_2018(TensorflowTransformer):
    """
    InceptionResnet v2 model trained in 2018 using the CasiaWebFace dataset in the context of the work:

    Freitas Pereira, Tiago, André Anjos, and Sébastien Marcel. "Heterogeneous face recognition using domain specific units." IEEE Transactions on Information Forensics and Security 14.7 (2018): 1803-1816.

    """

    def __init__(self, memory_demanding=False, **kwargs):

        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/inceptionresnetv2_casia_centerloss_2018.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/inceptionresnetv2_casia_centerloss_2018.tar.gz",
        ]

        filename = get_file(
            "inceptionresnetv2_casia_centerloss_2018.tar.gz",
            urls,
            cache_subdir="data/tensorflow/inceptionresnetv2_casia_centerloss_2018",
            file_hash="1e0b62e45430a8d7516d7a6101a24c40",
            extract=True,
        )
        checkpoint_path = os.path.dirname(filename)

        super(InceptionResnetv2_Casia_CenterLoss_2018, self).__init__(
            checkpoint_path,
            preprocessor=tf.image.per_image_standardization,
            memory_demanding=memory_demanding,
            **kwargs,
        )


class InceptionResnetv1_Casia_CenterLoss_2018(TensorflowTransformer):
    """
    InceptionResnet v1 model trained in 2018 using the CasiaWebFace dataset in the context of the work:

    Freitas Pereira, Tiago, André Anjos, and Sébastien Marcel. "Heterogeneous face recognition using domain specific units." IEEE Transactions on Information Forensics and Security 14.7 (2018): 1803-1816.

    """

    def __init__(self, memory_demanding=False, **kwargs):

        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/inceptionresnetv1_casia_centerloss_2018.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/inceptionresnetv1_casia_centerloss_2018.tar.gz",
        ]

        filename = get_file(
            "inceptionresnetv1_casia_centerloss_2018.tar.gz",
            urls,
            cache_subdir="data/tensorflow/inceptionresnetv1_casia_centerloss_2018",
            file_hash="6601e6f6840ae863c7daf31a7c6b9a27",
            extract=True,
        )
        checkpoint_path = os.path.dirname(filename)

        super(InceptionResnetv1_Casia_CenterLoss_2018, self).__init__(
            checkpoint_path,
            preprocessor=tf.image.per_image_standardization,
            memory_demanding=memory_demanding,
            **kwargs,
        )


class InceptionResnetv1_MsCeleb_CenterLoss_2018(TensorflowTransformer):
    """
    InceptionResnet v1 model trained in 2018 using the MsCeleb dataset in the context of the work:

    Freitas Pereira, Tiago, André Anjos, and Sébastien Marcel. "Heterogeneous face recognition using domain specific units." IEEE Transactions on Information Forensics and Security 14.7 (2018): 1803-1816.

    """

    def __init__(self, memory_demanding=False, **kwargs):

        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/inceptionresnetv1_msceleb_centerloss_2018.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/inceptionresnetv1_msceleb_centerloss_2018.tar.gz",
        ]

        filename = get_file(
            "inceptionresnetv1_msceleb_centerloss_2018.tar.gz",
            urls,
            cache_subdir="data/tensorflow/inceptionresnetv1_msceleb_centerloss_2018",
            file_hash="1ca0149619e4e9320a927ea65b2b5521",
            extract=True,
        )
        checkpoint_path = os.path.dirname(filename)

        super(InceptionResnetv1_MsCeleb_CenterLoss_2018, self).__init__(
            checkpoint_path,
            preprocessor=tf.image.per_image_standardization,
            memory_demanding=memory_demanding,
            **kwargs,
        )


class FaceNetSanderberg_20170512_110547(TensorflowTransformer):
    """
    Wrapper for the free FaceNet from David Sanderberg model 20170512_110547:
    https://github.com/davidsandberg/facenet

    And for a preprocessor you can use::

        from bob.bio.face.preprocessor import FaceCrop
        # This is the size of the image that this model expects
        CROPPED_IMAGE_HEIGHT = 160
        CROPPED_IMAGE_WIDTH = 160
        # eye positions for frontal images
        RIGHT_EYE_POS = (46, 53)
        LEFT_EYE_POS = (46, 107)
        # Crops the face using eye annotations
        preprocessor = FaceCrop(
            cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
            cropped_positions={'leye': LEFT_EYE_POS, 'reye': RIGHT_EYE_POS},
            color_channel='rgb'
        )
    """

    def __init__(self, memory_demanding=False, **kwargs):
        urls = [
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/facenet_sanderberg_20170512_110547.tar.gz"
        ]

        filename = get_file(
            "facenet_sanderberg_20170512_110547.tar.gz",
            urls,
            cache_subdir="data/tensorflow/facenet_sanderberg_20170512_110547",
            file_hash="734d1c997c10acdcdffc79fb51a2e715",
            extract=True,
        )
        checkpoint_path = os.path.dirname(filename)

        super(FaceNetSanderberg_20170512_110547, self).__init__(
            checkpoint_path,
            tf.image.per_image_standardization,
            memory_demanding=memory_demanding,
            **kwargs,
        )


class Resnet50_MsCeleb_ArcFace_2021(TensorflowTransformer):
    """
    Resnet50 Backbone trained with the MSCeleb 1M database.

    The bottleneck layer (a.k.a embedding) has 512d.

    The configuration file used to trained is:

    .. warning::
        This configuration file might change in future releases

    ```yaml
    batch-size: 128
    face-size: 112
    face-output_size: 112
    n-classes: 85742


    # Backbone
    backbone: 'resnet50'
    head: 'arcface'
    s: 10
    bottleneck: 512
    m: 0.5

    # Training parameters
    solver: "sgd"
    lr: 0.1
    dropout-rate: 0.5
    epochs: 500


    train-tf-record-path: "<PATH>"
    validation-tf-record-path: "<PATH>"

    ```


    """

    def __init__(self, memory_demanding=False, **kwargs):
        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/resnet50-msceleb-arcface_2021-48ec5cb8.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/resnet50-msceleb-arcface_2021-48ec5cb8.tar.gz",
        ]

        filename = get_file(
            "resnet50-msceleb-arcface_2021-48ec5cb8.tar.gz",
            urls,
            cache_subdir="data/tensorflow/resnet50-msceleb-arcface_2021-48ec5cb8",
            file_hash="17946f121af5ddd18c637c4620e54da6",
            extract=True,
        )
        checkpoint_path = os.path.dirname(filename)

        super(Resnet50_MsCeleb_ArcFace_2021, self).__init__(
            checkpoint_path,
            preprocessor=lambda X: X / 255.0,
            memory_demanding=memory_demanding,
            **kwargs,
        )


class Resnet50_MsCeleb_ArcFace_20210521(TensorflowTransformer):
    """
    Resnet50 Backbone trained with the MSCeleb 1M database. The bottleneck layer (a.k.a embedding) has 512d.

    The difference from this one to :any:`Resnet50_MsCeleb_ArcFace_2021` is the MSCeleb version used to train it.
    This one uses 100% of the data pruned from annotators.


    The configuration file used to trained is:

    .. warning::
        This configuration file might change in future releases


    ```yaml
    batch-size: 128
    face-size: 112
    face-output_size: 112
    n-classes: 83009


    # Backbone
    backbone: 'resnet50'
    head: 'arcface'
    s: 30
    bottleneck: 512
    m: 0.5

    # Training parameters
    solver: "sgd"
    lr: 0.1
    dropout-rate: 0.5
    epochs: 300


    train-tf-record-path: "<PATH>"
    validation-tf-record-path: "<PATH>"

    ```


    """

    def __init__(self, memory_demanding=False, **kwargs):

        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/resnet50-msceleb-arcface_20210521-e9bc085c.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/resnet50-msceleb-arcface_20210521-e9bc085c.tar.gz",
        ]

        filename = get_file(
            "resnet50-msceleb-arcface_20210521-e9bc085c.tar.gz",
            urls,
            cache_subdir="data/tensorflow/resnet50-msceleb-arcface_20210521-801991f0",
            file_hash="e33090eea4951ce80be4620a0dac680d",
            extract=True,
        )
        checkpoint_path = os.path.dirname(filename)

        super(Resnet50_MsCeleb_ArcFace_20210521, self).__init__(
            checkpoint_path,
            preprocessor=lambda X: X / 255.0,
            memory_demanding=memory_demanding,
            **kwargs,
        )


class Resnet101_MsCeleb_ArcFace_20210521(TensorflowTransformer):
    """
    Resnet101 Backbone trained with the MSCeleb 1M database. The bottleneck layer (a.k.a embedding) has 512d.

    The difference from this one to :any:`Resnet101_MsCeleb_ArcFace_2021` is the MSCeleb version used to train it.
    This one uses 100% of the data pruned from annotators.


    The configuration file used to trained is:

    .. warning::
        This configuration file might change in future releases


    ```yaml
    batch-size: 128
    face-size: 112
    face-output_size: 112
    n-classes: 83009


    # Backbone
    backbone: 'resnet50'
    head: 'arcface'
    s: 30
    bottleneck: 512
    m: 0.5

    # Training parameters
    solver: "sgd"
    lr: 0.1
    dropout-rate: 0.5
    epochs: 300


    train-tf-record-path: "<PATH>"
    validation-tf-record-path: "<PATH>"

    ```


    """

    def __init__(self, memory_demanding=False, **kwargs):

        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/resnet101-msceleb-arcface_20210521.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/resnet101-msceleb-arcface_20210521.tar.gz",
        ]

        filename = get_file(
            "resnet101-msceleb-arcface_20210521.tar.gz",
            urls,
            cache_subdir="data/tensorflow/resnet101-msceleb-arcface_20210521",
            file_hash="c1b2124cb69186ff965f7e818f9f8641",
            extract=True,
        )
        checkpoint_path = os.path.dirname(filename)

        super(Resnet101_MsCeleb_ArcFace_20210521, self).__init__(
            checkpoint_path,
            preprocessor=lambda X: X / 255.0,
            memory_demanding=memory_demanding,
            **kwargs,
        )


class IResnet50_MsCeleb_ArcFace_20210623(TensorflowTransformer):
    """
    IResnet50 Backbone trained with the MSCeleb 1M database. The bottleneck layer (a.k.a embedding) has 512d.

    The complete code to reproduce this model is in the (private) repository:
    bob.project.hardening/-/commit/9ac25c0a17c9628b7a99e84217cd7c680f1a3e1e
    but you can reproduce it using
    https://gitlab.idiap.ch/bob/bob.bio.face/-/blob/eed9276c7c1306c2ccfe290a0149ade3a80d247a/cnn_training/arcface_large_batch.py
    script and the following configuration::

        CONFIG = {
            "n-workers": 8,
            "batch-size": 256,
            "n-train-samples-per-epoch": 256_000 * 1,
            "real-n-train-samples": 985702,
            "shuffle-buffer": int(1e6),
            "face-size": 126,
            "face-output_size": 112,
            "n-classes": 83009,
            "backbone": "resnet50_large_batch",
            "use-l2-regularizer": False,
            "batch-norm-decay": 0.9,
            "batch-norm-epsilon": 1e-5,
            "head": "arcface",
            "s": 30,
            "bottleneck": 512,
            "m": 0.5,
            "dropout-rate": 0.0,
            "learning-rate-schedule": "none",
            "train-tf-record-path": "/face-tfrecords/126x126/msceleb_facecrop/*.tfrecords",
            "validation-tf-record-path": "/face-tfrecords/126x126/lfw_sharded/*.tfrecords",
            "checkpoint-path": "/temp/hardening/arcface_sgd_prelu/w8_b1000_fp16_drp0",
            "pre-train": False,
            "epochs": 6000,
        }
        strategy_fn = "multi-worker-mirrored-strategy"
        mixed_precision_policy = "mixed_float16"
        initial_lr = 0.1 / 512 * CONFIG["batch-size"] * CONFIG["n-workers"]
        real_n_steps_per_epoch = CONFIG["real-n-train-samples"] / (CONFIG["batch-size"] * CONFIG["n-workers"])
        params = {
            "optimizer": {
                "type": "sgdw",
                "sgdw": {
                    "momentum": min(0.9 * initial_lr, 0.999),
                    "nesterov": False,
                    "weight_decay": 5e-4,
                },
            },
            "learning_rate": {
                "type": "stepwise",
                "stepwise": {
                    "boundaries": [int(i * real_n_steps_per_epoch) for i in [11, 17, 22]],
                    "values": [initial_lr / (10 ** i) for i in range(0, 4)],
                },
            },
        }

    The tensorboard logs can be found in: https://tensorboard.dev/experiment/6bBn0ya3SeilJ2elcZZoSg
    The model at epoch 90 is used.
    """

    def __init__(self, memory_demanding=False, **kwargs):

        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/arcface_iresnet50_msceleb_idiap-089640d2.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/arcface_iresnet50_msceleb_idiap-089640d2.tar.gz",
        ]

        filename = get_file(
            "arcface_iresnet50_msceleb_idiap-089640d2.tar.gz",
            urls,
            cache_subdir="data/tensorflow/arcface_iresnet50_msceleb_idiap-089640d2",
            file_hash="089640d2",
            extract=True,
        )
        checkpoint_path = os.path.dirname(filename)

        super().__init__(
            checkpoint_path,
            preprocessor=lambda X: X / 255.0,
            memory_demanding=memory_demanding,
            **kwargs,
        )


class IResnet100_MsCeleb_ArcFace_20210623(TensorflowTransformer):
    """
    IResnet100 Backbone trained with the MSCeleb 1M database. The bottleneck layer (a.k.a embedding) has 512d.

    The complete code to reproduce this model is in the (private) repository:
    bob.project.hardening/-/commit/b162ca60d26fcf8a93f6767f5b5a026a406c1076
    but you can reproduce it using
    https://gitlab.idiap.ch/bob/bob.bio.face/-/blob/eed9276c7c1306c2ccfe290a0149ade3a80d247a/cnn_training/arcface_large_batch.py
    script and the following configuration::

        CONFIG = {
            "n-workers": 8,
            "batch-size": 128,
            "n-train-samples-per-epoch": 256_000 * 1,
            "real-n-train-samples": 985702,
            "shuffle-buffer": int(1e5),
            "face-size": 126,
            "face-output_size": 112,
            "n-classes": 83009,
            "backbone": "iresnet100",
            "use-l2-regularizer": False,
            "batch-norm-decay": 0.9,
            "batch-norm-epsilon": 1e-5,
            "head": "arcface",
            "s": 30,
            "bottleneck": 512,
            "m": 0.5,
            "dropout-rate": 0.0,
            "learning-rate-schedule": "none",
            "train-tf-record-path": "/face-tfrecords/126x126/msceleb_facecrop/*.tfrecords",
            "validation-tf-record-path": "/face-tfrecords/126x126/lfw_sharded/*.tfrecords",
            "checkpoint-path": "/temp/hardening/arcface_sgd_prelu/i100_w8_b128_fp16_drp0",
            "pre-train": False,
            "epochs": 6000,
        }
        strategy_fn = "multi-worker-mirrored-strategy"
        mixed_precision_policy = "mixed_float16"
        initial_lr = 0.1 / 512 * CONFIG["batch-size"] * CONFIG["n-workers"]
        real_n_steps_per_epoch = CONFIG["real-n-train-samples"] / (CONFIG["batch-size"] * CONFIG["n-workers"])
        params = {
            "optimizer": {
                "type": "sgdw",
                "sgdw": {
                    "momentum": min(0.9 * initial_lr, 0.999),
                    "nesterov": False,
                    "weight_decay": 5e-4,
                },
            },
            "learning_rate": {
                # with ReduceLROnPlateau callback
                "type": "constant",
                "constant": {
                    "learning_rate": initial_lr,
                }
            },
        }

    The tensorboard logs can be found in: https://tensorboard.dev/experiment/HYJTPiowRMa36VZHDLJqdg/
    The model is saved based on best ``epoch_embeddings_embedding_accuracy``, epoch 51
    """

    def __init__(self, memory_demanding=False):

        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/arcface_iresnet100_msceleb_idiap-1b22d544.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/arcface_iresnet100_msceleb_idiap-1b22d544.tar.gz",
        ]

        filename = get_file(
            "arcface_iresnet100_msceleb_idiap-1b22d544.tar.gz",
            urls,
            cache_subdir="data/tensorflow/arcface_iresnet100_msceleb_idiap-1b22d544",
            file_hash="1b22d544",
            extract=True,
        )
        checkpoint_path = os.path.dirname(filename)

        super().__init__(
            checkpoint_path,
            preprocessor=lambda X: X / 255.0,
            memory_demanding=memory_demanding,
        )


class Resnet50_VGG2_ArcFace_2021(TensorflowTransformer):
    """
    Resnet50 Backbone trained with the VGG2 database.

    The bottleneck layer (a.k.a embedding) has 512d.

    The configuration file used to trained is:

    .. warning::
        This configuration file might change in future releases

    ```yaml
    batch-size: 128
    face-size: 112
    face-output_size: 112
    n-classes: 8631


    # Backbone
    backbone: 'resnet50'
    head: 'arcface'
    s: 64
    bottleneck: 512
    m: 0.5

    # Training parameters
    solver: "sgd"
    lr: 0.1
    dropout-rate: 0.5
    epochs: 1047


    train-tf-record-path: "<PATH>"
    validation-tf-record-path: "<PATH>"

    ```


    """

    def __init__(self, memory_demanding=False, **kwargs):
        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/resnet50_vgg2_arcface_2021.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/resnet50_vgg2_arcface_2021.tar.gz",
        ]

        filename = get_file(
            "resnet50_vgg2_arcface_2021.tar.gz",
            urls,
            cache_subdir="data/tensorflow/resnet50_vgg2_arcface_2021",
            file_hash="64f89c8cb55e7a0d9c7e13ff412b6a13",
            extract=True,
        )
        checkpoint_path = os.path.dirname(filename)

        super(Resnet50_VGG2_ArcFace_2021, self).__init__(
            checkpoint_path,
            preprocessor=lambda X: X / 255.0,
            memory_demanding=memory_demanding,
            **kwargs,
        )

    def inference(self, X):
        if self.preprocessor is not None:
            X = self.preprocessor(tf.cast(X, "float32"))

        prelogits = self.model.predict_on_batch(X)
        embeddings = tf.math.l2_normalize(prelogits, axis=-1)
        return embeddings


class MobileNetv2_MsCeleb_ArcFace_2021(TensorflowTransformer):
    """
    MobileNet Backbone trained with the MSCeleb 1M database.

    The bottleneck layer (a.k.a embedding) has 512d.

    The configuration file used to trained is:

    .. warning::
        This configuration file might change in future releases

    ```yaml
    batch-size: 128
    face-size: 112
    face-output_size: 112
    n-classes: 85742


    # Backbone
    backbone: 'mobilenet-v2'
    head: 'arcface'
    s: 10
    bottleneck: 512
    m: 0.5

    # Training parameters
    solver: "sgd"
    lr: 0.01
    dropout-rate: 0.5
    epochs: 500


    train-tf-record-path: "<PATH>"
    validation-tf-record-path: "<PATH>"

    ```


    """

    def __init__(self, memory_demanding=False, **kwargs):

        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/mobilenet-v2-msceleb-arcface-2021-e012cb66.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/mobilenet-v2-msceleb-arcface-2021-e012cb66.tar.gz",
        ]

        filename = get_file(
            "mobilenet-v2-msceleb-arcface-2021-e012cb66.tar.gz",
            urls,
            cache_subdir="data/tensorflow/mobilenet-v2-msceleb-arcface-2021-e012cb66",
            file_hash="dd1399b86f01725c7b07b480b703e02a",
            extract=True,
        )
        checkpoint_path = os.path.dirname(filename)

        super(MobileNetv2_MsCeleb_ArcFace_2021, self).__init__(
            checkpoint_path,
            preprocessor=lambda X: X / 255.0,
            memory_demanding=memory_demanding,
            **kwargs,
        )


def facenet_template(embedding, annotation_type, fixed_positions=None):
    """
    Facenet baseline template.
    This one will crop the face at :math:`160 \\times 160`

    Parameters
    ----------

      embedding: obj
         Transformer that takes a cropped face and extract the embeddings

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image
    """
    # DEFINE CROPPING
    cropped_image_size = (160, 160)

    if annotation_type == "eyes-center" or annotation_type == "bounding-box":
        # Hard coding eye positions for backward consistency
        # cropped_positions = {
        cropped_positions = dnn_default_cropping(
            cropped_image_size, annotation_type="eyes-center"
        )
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

    # ASSEMBLE TRANSFORMER
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


def resnet_template(embedding, annotation_type, fixed_positions=None):
    # DEFINE CROPPING
    # cropped_image_size = (112, 112)
    # if annotation_type == "eyes-center":
    #    # Hard coding eye positions for backward consistency
    #    cropped_positions = cropped_positions_arcface()
    # else:
    #    cropped_positions = dnn_default_cropping(cropped_image_size, annotation_type)
    # DEFINE CROPPING
    cropped_image_size = (112, 112)
    if annotation_type == "eyes-center" or annotation_type == "bounding-box":
        # Hard coding eye positions for backward consistency
        # cropped_positions = {
        cropped_positions = cropped_positions_arcface()
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


def resnet50_msceleb_arcface_2021(
    annotation_type, fixed_positions=None, memory_demanding=False
):
    """
    Get the Resnet50 pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`Resnet50_MsCeleb_ArcFace_2021` to extract the features

    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return resnet_template(
        embedding=Resnet50_MsCeleb_ArcFace_2021(
            memory_demanding=memory_demanding
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def resnet50_msceleb_arcface_20210521(
    annotation_type, fixed_positions=None, memory_demanding=False
):
    """
    Get the Resnet50 pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`Resnet50_MsCeleb_ArcFace_20210521` to extract the features

    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return resnet_template(
        embedding=Resnet50_MsCeleb_ArcFace_20210521(
            memory_demanding=memory_demanding
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def resnet101_msceleb_arcface_20210521(
    annotation_type, fixed_positions=None, memory_demanding=False
):
    """
    Get the Resnet50 pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`Resnet50_MsCeleb_ArcFace_20210521` to extract the features

    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return resnet_template(
        embedding=Resnet101_MsCeleb_ArcFace_20210521(
            memory_demanding=memory_demanding
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def iresnet50_msceleb_arcface_20210623(
    annotation_type, fixed_positions=None, memory_demanding=False
):
    """
    Get the iresnet50 pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`IResnet50_MsCeleb_ArcFace_20210623` to extract the features

    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """
    return resnet_template(
        embedding=IResnet50_MsCeleb_ArcFace_20210623(
            memory_demanding=memory_demanding
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def iresnet100_msceleb_arcface_20210623(
    annotation_type, fixed_positions=None, memory_demanding=False
):
    """
    Get the iresnet100 pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`IResnet100_MsCeleb_ArcFace_20210623` to extract the features

    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """
    return resnet_template(
        embedding=IResnet100_MsCeleb_ArcFace_20210623(
            memory_demanding=memory_demanding
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def resnet50_vgg2_arcface_2021(
    annotation_type, fixed_positions=None, memory_demanding=False
):
    """
    Get the Resnet50 pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`Resnet50_VGG2_ArcFace_2021` to extract the features

    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return resnet_template(
        embedding=Resnet50_VGG2_ArcFace_2021(memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def mobilenetv2_msceleb_arcface_2021(
    annotation_type, fixed_positions=None, memory_demanding=False
):
    """
    Get the MobileNet pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`MobileNetv2_MsCeleb_ArcFace_2021` to extract the features

    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return resnet_template(
        embedding=MobileNetv2_MsCeleb_ArcFace_2021(
            memory_demanding=memory_demanding
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def facenet_sanderberg_20170512_110547(
    annotation_type, fixed_positions=None, memory_demanding=False
):
    """
    Get the Facenet pipeline which will crop the face :math:`160 \\times 160` and
    use the :py:class:`FaceNetSanderberg_20170512_110547` to extract the features

    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return facenet_template(
        embedding=FaceNetSanderberg_20170512_110547(
            memory_demanding=memory_demanding
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def inception_resnet_v1_casia_centerloss_2018(
    annotation_type, fixed_positions=None, memory_demanding=False
):
    """
    Get the Inception Resnet v1 pipeline which will crop the face :math:`160 \\times 160` and
    use the :py:class:`InceptionResnetv1_Casia_CenterLoss_2018` to extract the features

    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return facenet_template(
        embedding=InceptionResnetv1_Casia_CenterLoss_2018(
            memory_demanding=memory_demanding
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def inception_resnet_v2_casia_centerloss_2018(
    annotation_type, fixed_positions=None, memory_demanding=False
):
    """
    Get the Inception Resnet v2 pipeline which will crop the face :math:`160 \\times 160` and
    use the :py:class:`InceptionResnetv2_Casia_CenterLoss_2018` to extract the features

    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return facenet_template(
        embedding=InceptionResnetv2_Casia_CenterLoss_2018(
            memory_demanding=memory_demanding
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def inception_resnet_v1_msceleb_centerloss_2018(
    annotation_type, fixed_positions=None, memory_demanding=False
):
    """
    Get the Inception Resnet v1 pipeline which will crop the face :math:`160 \\times 160` and
    use the :py:class:`InceptionResnetv1_MsCeleb_CenterLoss_2018` to extract the features

    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return facenet_template(
        embedding=InceptionResnetv1_MsCeleb_CenterLoss_2018(
            memory_demanding=memory_demanding
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def inception_resnet_v2_msceleb_centerloss_2018(
    annotation_type, fixed_positions=None, memory_demanding=False
):
    """
    Get the Inception Resnet v2 pipeline which will crop the face :math:`160 \\times 160` and
    use the :py:class:`InceptionResnetv2_MsCeleb_CenterLoss_2018` to extract the features

    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return facenet_template(
        embedding=InceptionResnetv2_MsCeleb_CenterLoss_2018(
            memory_demanding=memory_demanding
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )
