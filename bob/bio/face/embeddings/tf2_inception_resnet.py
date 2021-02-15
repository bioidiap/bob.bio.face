import tensorflow as tf
from bob.learn.tensorflow.utils.image import to_channels_last
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array

from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from bob.extension import rc
from functools import partial
import pkg_resources
import os
from bob.bio.face.embeddings import download_model
import numpy as np


def sanderberg_rescaling():
    # FIXED_STANDARDIZATION from https://github.com/davidsandberg/facenet
    # [-0.99609375, 0.99609375]
    preprocessor = preprocessing.Rescaling(scale=1 / 128, offset=-127.5 / 128)
    return preprocessor


class TransformTensorflow(TransformerMixin, BaseEstimator):
    """
    Base Transformer for Tensorflow architectures.

    Szegedy, Christian, et al. "Inception-v4, inception-resnet and the impact of residual connections on learning." arXiv preprint arXiv:1602.07261 (2016).

    Parameters
    ----------

    checkpoint_path: str
       Path containing the checkpoint

    preprocessor:
        Preprocessor function

    memory_demanding bool
        If `True`, the `transform` method will run one sample at the time.
        This is useful when there is not enough memory available to forward big chucks of data.
    """

    def __init__(
        self, checkpoint_path, preprocessor=None, memory_demanding=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.preprocessor = preprocessor
        self.memory_demanding = memory_demanding

    def load_model(self):
        self.model = tf.keras.models.load_model(self.checkpoint_path)

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
            return np.array([_transform(x[None, ...]) for x in X])
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
        return {"stateless": True, "requires_fit": False}

    def __del__(self):
        self.model = None


class InceptionResnetv2_MsCeleb_CenterLoss_2018(TransformTensorflow):
    """
    InceptionResnet v2 model trained in 2018 using the MSCeleb dataset in the context of the work:

    Freitas Pereira, Tiago, André Anjos, and Sébastien Marcel. "Heterogeneous face recognition using domain specific units." IEEE Transactions on Information Forensics and Security 14.7 (2018): 1803-1816.

    """

    def __init__(self, memory_demanding=False):
        internal_path = pkg_resources.resource_filename(
            __name__, os.path.join("data", "inceptionresnetv2_msceleb_centerloss_2018"),
        )

        checkpoint_path = (
            internal_path
            if rc["bob.bio.face.models.InceptionResnetv2_MsCeleb_CenterLoss_2018"]
            is None
            else rc["bob.bio.face.models.InceptionResnetv2_MsCeleb_CenterLoss_2018"]
        )

        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/inceptionresnetv2_msceleb_centerloss_2018.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/inceptionresnetv2_msceleb_centerloss_2018.tar.gz",
        ]

        download_model(
            checkpoint_path, urls, "inceptionresnetv2_msceleb_centerloss_2018.tar.gz"
        )

        super(InceptionResnetv2_MsCeleb_CenterLoss_2018, self).__init__(
            checkpoint_path,
            preprocessor=tf.image.per_image_standardization,
            memory_demanding=memory_demanding,
        )


class InceptionResnetv2_Casia_CenterLoss_2018(TransformTensorflow):
    """
    InceptionResnet v2 model trained in 2018 using the CasiaWebFace dataset in the context of the work:

    Freitas Pereira, Tiago, André Anjos, and Sébastien Marcel. "Heterogeneous face recognition using domain specific units." IEEE Transactions on Information Forensics and Security 14.7 (2018): 1803-1816.

    """

    def __init__(self, memory_demanding=False):
        internal_path = pkg_resources.resource_filename(
            __name__, os.path.join("data", "inceptionresnetv2_casia_centerloss_2018"),
        )

        checkpoint_path = (
            internal_path
            if rc["bob.bio.face.models.InceptionResnetv2_Casia_CenterLoss_2018"] is None
            else rc["bob.bio.face.models.InceptionResnetv2_Casia_CenterLoss_2018"]
        )

        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/inceptionresnetv2_casia_centerloss_2018.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/inceptionresnetv2_casia_centerloss_2018.tar.gz",
        ]

        download_model(
            checkpoint_path, urls, "inceptionresnetv2_casia_centerloss_2018.tar.gz"
        )

        super(InceptionResnetv2_Casia_CenterLoss_2018, self).__init__(
            checkpoint_path,
            preprocessor=tf.image.per_image_standardization,
            memory_demanding=memory_demanding,
        )


class InceptionResnetv1_Casia_CenterLoss_2018(TransformTensorflow):
    """
    InceptionResnet v1 model trained in 2018 using the CasiaWebFace dataset in the context of the work:

    Freitas Pereira, Tiago, André Anjos, and Sébastien Marcel. "Heterogeneous face recognition using domain specific units." IEEE Transactions on Information Forensics and Security 14.7 (2018): 1803-1816.

    """

    def __init__(self, memory_demanding=False):
        internal_path = pkg_resources.resource_filename(
            __name__, os.path.join("data", "inceptionresnetv1_casia_centerloss_2018"),
        )

        checkpoint_path = (
            internal_path
            if rc["bob.bio.face.models.InceptionResnetv1_Casia_CenterLoss_2018"] is None
            else rc["bob.bio.face.models.InceptionResnetv1_Casia_CenterLoss_2018"]
        )

        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/inceptionresnetv1_casia_centerloss_2018.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/inceptionresnetv1_casia_centerloss_2018.tar.gz",
        ]

        download_model(
            checkpoint_path, urls, "inceptionresnetv1_casia_centerloss_2018.tar.gz"
        )

        super(InceptionResnetv1_Casia_CenterLoss_2018, self).__init__(
            checkpoint_path,
            preprocessor=tf.image.per_image_standardization,
            memory_demanding=memory_demanding,
        )


class InceptionResnetv1_MsCeleb_CenterLoss_2018(TransformTensorflow):
    """
    InceptionResnet v1 model trained in 2018 using the MsCeleb dataset in the context of the work:

    Freitas Pereira, Tiago, André Anjos, and Sébastien Marcel. "Heterogeneous face recognition using domain specific units." IEEE Transactions on Information Forensics and Security 14.7 (2018): 1803-1816.

    """

    def __init__(self, memory_demanding=False):
        internal_path = pkg_resources.resource_filename(
            __name__, os.path.join("data", "inceptionresnetv1_msceleb_centerloss_2018"),
        )

        checkpoint_path = (
            internal_path
            if rc["bob.bio.face.models.InceptionResnetv1_MsCeleb_CenterLoss_2018"]
            is None
            else rc["bob.bio.face.models.InceptionResnetv1_MsCeleb_CenterLoss_2018"]
        )

        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/inceptionresnetv1_msceleb_centerloss_2018.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/inceptionresnetv1_msceleb_centerloss_2018.tar.gz",
        ]

        download_model(
            checkpoint_path, urls, "inceptionresnetv1_msceleb_centerloss_2018.tar.gz"
        )

        super(InceptionResnetv1_MsCeleb_CenterLoss_2018, self).__init__(
            checkpoint_path,
            preprocessor=tf.image.per_image_standardization,
            memory_demanding=memory_demanding,
        )


class FaceNetSanderberg_20170512_110547(TransformTensorflow):
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

    def __init__(self, memory_demanding=False):
        internal_path = pkg_resources.resource_filename(
            __name__, os.path.join("data", "facenet_sanderberg_20170512_110547"),
        )

        checkpoint_path = (
            internal_path
            if rc["bob.bio.face.models.facenet_sanderberg_20170512_110547"] is None
            else rc["bob.bio.face.models.facenet_sanderberg_20170512_110547"]
        )

        urls = [
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/facenet_sanderberg_20170512_110547.tar.gz"
        ]

        download_model(
            checkpoint_path, urls, "facenet_sanderberg_20170512_110547.tar.gz"
        )

        super(FaceNetSanderberg_20170512_110547, self).__init__(
            checkpoint_path,
            tf.image.per_image_standardization,
            memory_demanding=memory_demanding,
        )

