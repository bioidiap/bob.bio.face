import bob.bio.face
import numpy as np


def test_facenet():
    from bob.bio.face.transformers import FaceNetSanderberg

    np.random.seed(10)

    transformer = FaceNetSanderberg()
    data = np.random.rand(3, 160, 160).astype("uint8")
    output = transformer.transform(data)
    assert output.size == 128, output.shape


def test_idiap_inceptionv2_msceleb():
    from bob.bio.face.transformers import InceptionResnetv2_MsCeleb

    np.random.seed(10)
    transformer = InceptionResnetv2_MsCeleb()
    data = np.random.rand(3, 160, 160).astype("uint8")
    output = transformer.transform(data)
    assert output.size == 128, output.shape


def test_idiap_inceptionv2_casia():
    from bob.bio.face.transformers import InceptionResnetv2_CasiaWebFace

    np.random.seed(10)
    transformer = InceptionResnetv2_CasiaWebFace()
    data = np.random.rand(3, 160, 160).astype("uint8")
    output = transformer.transform(data)
    assert output.size == 128, output.shape


def test_idiap_inceptionv1_msceleb():
    from bob.bio.face.transformers import InceptionResnetv1_MsCeleb

    np.random.seed(10)
    transformer = InceptionResnetv1_MsCeleb()
    data = np.random.rand(3, 160, 160).astype("uint8")
    output = transformer.transform(data)
    assert output.size == 128, output.shape


def test_idiap_inceptionv1_casia():
    from bob.bio.face.transformers import InceptionResnetv1_CasiaWebFace

    np.random.seed(10)
    transformer = InceptionResnetv1_CasiaWebFace()
    data = np.random.rand(3, 160, 160).astype("uint8")
    output = transformer.transform(data)
    assert output.size == 128, output.shape
