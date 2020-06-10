#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import os
from sklearn.base import TransformerMixin, BaseEstimator
from .tensorflow_compat_v1 import TensorflowCompatV1


class InceptionResnetv2_MsCeleb(TensorflowCompatV1):
    """
    Inception Restnet v2 model from https://gitlab.idiap.ch/bob/bob.learn.tensorflow/-/blob/1e40a68bfbbb3dd8813c48d50b2f23ff7a399956/bob/learn/tensorflow/network/InceptionResnetV2.py

    This model was trained using the MsCeleb 1M dataset

    The input shape of this model is :math:`3 \times 160 \times 160`
    The output embedding is :math:`n \times 128`, where :math:`n` is the number of samples

    """

    def __init__(self):
        from bob.learn.tensorflow.network import inception_resnet_v2_batch_norm

        bob_rc_variable = "bob.bio.face.idiap_inception_resnet_v2_path"
        urls = ["https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/inception-v2_batchnorm_rgb.tar.gz"]
        model_subdirectory = "idiap_inception_resnet_v2_path"

        
        checkpoint_filename = self.get_modelpath(bob_rc_variable, model_subdirectory)
        self.download_model(checkpoint_filename, urls)

        input_shape = (1, 160, 160, 3)
        architecture_fn = inception_resnet_v2_batch_norm

        super().__init__(checkpoint_filename, input_shape, architecture_fn)


class InceptionResnetv2_CasiaWebFace(TensorflowCompatV1):
    """
    Inception Restnet v2 model from https://gitlab.idiap.ch/bob/bob.learn.tensorflow/-/blob/1e40a68bfbbb3dd8813c48d50b2f23ff7a399956/bob/learn/tensorflow/network/InceptionResnetV2.py

    This model was trained using the Casia WebFace

    The input shape of this model is :math:`3 \times 160 \times 160`
    The output embedding is :math:`n \times 128`, where :math:`n` is the number of samples

    """

    def __init__(self):
        """Loads the tensorflow model
        """
        from bob.learn.tensorflow.network import inception_resnet_v2_batch_norm

        bob_rc_variable = "bob.bio.face.idiap_inception_resnet_v2_casiawebface_path"
        urls = ["https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/idiap_inception_resnet_v2_casiawebface_path.tar.gz"]
        model_subdirectory = "idiap_inception_resnet_v2_casiawebface_path"


        checkpoint_filename = self.get_modelpath(bob_rc_variable, model_subdirectory)
        self.download_model(checkpoint_filename, urls)

        input_shape = (1, 160, 160, 3)
        architecture_fn = inception_resnet_v2_batch_norm

        super().__init__(checkpoint_filename, input_shape, architecture_fn)


class InceptionResnetv1_MsCeleb(TensorflowCompatV1):
    """
    Inception Restnet v1 model from https://gitlab.idiap.ch/bob/bob.learn.tensorflow/-/blob/1e40a68bfbbb3dd8813c48d50b2f23ff7a399956/bob/learn/tensorflow/network/InceptionResnetV1.py

    This model was trained using the MsCeleb 1M dataset

    The input shape of this model is :math:`3 \times 160 \times 160`
    The output embedding is :math:`n \times 128`, where :math:`n` is the number of samples

    """

    def __init__(self):
        from bob.learn.tensorflow.network import inception_resnet_v1_batch_norm

        bob_rc_variable = "bob.bio.face.idiap_inception_resnet_v1_msceleb_path"
        urls = ["https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/idiap_inception_resnet_v1_msceleb_path.tar.gz"]
        model_subdirectory = "idiap_inception_resnet_v1_msceleb_path"

        
        checkpoint_filename = self.get_modelpath(bob_rc_variable, model_subdirectory)
        self.download_model(checkpoint_filename, urls)

        input_shape = (1, 160, 160, 3)
        architecture_fn = inception_resnet_v1_batch_norm

        super().__init__(checkpoint_filename, input_shape, architecture_fn)



class InceptionResnetv1_CasiaWebFace(TensorflowCompatV1):
    """
    Inception Restnet v1 model from https://gitlab.idiap.ch/bob/bob.learn.tensorflow/-/blob/1e40a68bfbbb3dd8813c48d50b2f23ff7a399956/bob/learn/tensorflow/network/InceptionResnetV1.py

    This model was trained using the Casia WebFace

    The input shape of this model is :math:`3 \times 160 \times 160`
    The output embedding is :math:`n \times 128`, where :math:`n` is the number of samples

    """

    def __init__(self):
        """Loads the tensorflow model
        """
        from bob.learn.tensorflow.network import inception_resnet_v1_batch_norm

        bob_rc_variable = "bob.bio.face.idiap_inception_resnet_v1_casiawebface_path"
        urls = ["https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/idiap_inception_resnet_v1_casiawebface_path.tar.gz"]
        model_subdirectory = "idiap_inception_resnet_v1_casiawebface_path"


        checkpoint_filename = self.get_modelpath(bob_rc_variable, model_subdirectory)
        self.download_model(checkpoint_filename, urls)

        input_shape = (1, 160, 160, 3)
        architecture_fn = inception_resnet_v1_batch_norm

        super().__init__(checkpoint_filename, input_shape, architecture_fn)
