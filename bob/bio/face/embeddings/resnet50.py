from bob.bio.face.embeddings import download_model


from .tf2_inception_resnet import TransformTensorflow
import pkg_resources
import os
from bob.extension import rc
import tensorflow as tf


class Resnet50_MsCeleb_ArcFace_2021(TransformTensorflow):
    """
    Resnet50 Backbone trained with the MSCeleb 1M database.

    The bottleneck layer (a.k.a embedding) has 512d.

    The configuration file used to trained is:

    ```yaml
    batch-size: 128
    face-size: 112
    face-output_size: 112
    n-classes: 85742


    ## Backbone
    backbone: 'resnet50'
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

    def __init__(self, memory_demanding=False):
        internal_path = pkg_resources.resource_filename(
            __name__, os.path.join("data", "resnet50_msceleb_arcface_2021"),
        )

        checkpoint_path = (
            internal_path
            if rc["bob.bio.face.models.resnet50_msceleb_arcface_2021"] is None
            else rc["bob.bio.face.models.resnet50_msceleb_arcface_2021"]
        )

        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/resnet50_msceleb_arcface_2021.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/resnet50_msceleb_arcface_2021.tar.gz",
        ]

        download_model(checkpoint_path, urls, "resnet50_msceleb_arcface_2021.tar.gz")

        super(Resnet50_MsCeleb_ArcFace_2021, self).__init__(
            checkpoint_path,
            preprocessor=lambda X: X / 255.0,
            memory_demanding=memory_demanding,
        )

    def inference(self, X):
        if self.preprocessor is not None:
            X = self.preprocessor(tf.cast(X, "float32"))

        prelogits = self.model.predict_on_batch(X)[0]
        embeddings = tf.math.l2_normalize(prelogits, axis=-1)
        return embeddings


class Resnet50_VGG2_ArcFace_2021(TransformTensorflow):
    """
    Resnet50 Backbone trained with the VGG2 database.

    The bottleneck layer (a.k.a embedding) has 512d.

    The configuration file used to trained is:

    ```yaml
    batch-size: 128
    face-size: 112
    face-output_size: 112
    n-classes: 8631


    ## Backbone
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

    def __init__(self, memory_demanding=False):
        internal_path = pkg_resources.resource_filename(
            __name__, os.path.join("data", "resnet50_vgg2_arcface_2021"),
        )

        checkpoint_path = (
            internal_path
            if rc["bob.bio.face.models.resnet50_vgg2_arcface_2021"] is None
            else rc["bob.bio.face.models.resnet50_vgg2_arcface_2021"]
        )

        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/resnet50_vgg2_arcface_2021.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/tf2/resnet50_vgg2_arcface_2021.tar.gz",
        ]

        download_model(checkpoint_path, urls, "resnet50_vgg2_arcface_2021.tar.gz")

        super(Resnet50_VGG2_ArcFace_2021, self).__init__(
            checkpoint_path,
            preprocessor=lambda X: X / 255.0,
            memory_demanding=memory_demanding,
        )

    def inference(self, X):
        if self.preprocessor is not None:
            X = self.preprocessor(tf.cast(X, "float32"))

        prelogits = self.model.predict_on_batch(X)
        embeddings = tf.math.l2_normalize(prelogits, axis=-1)
        return embeddings
