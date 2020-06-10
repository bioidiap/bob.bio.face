import bob.bio.face
import numpy as np
from bob.pipelines import Sample, wrap

def test_facenet():
    from bob.bio.face.embeddings import FaceNetSanderberg

    np.random.seed(10)

    transformer = FaceNetSanderberg()    
    # Raw data
    data = np.random.rand(3, 160, 160).astype("uint8")
    output = transformer.transform(data)
    assert output.size == 128, output.shape
    

    # Sample Batch
    sample = Sample(data)
    transformer_sample = wrap(["sample"], transformer)
    output = [s.data for s in transformer_sample.transform([sample])][0]

    assert output.size == 128, output.shape


def test_idiap_inceptionv2_msceleb():
    from bob.bio.face.embeddings import InceptionResnetv2_MsCeleb

    np.random.seed(10)
    transformer = InceptionResnetv2_MsCeleb()
    data = np.random.rand(3, 160, 160).astype("uint8")
    output = transformer.transform(data)
    assert output.size == 128, output.shape

    # Sample Batch
    sample = Sample(data)
    transformer_sample = wrap(["sample"], transformer)
    output = [s.data for s in transformer_sample.transform([sample])][0]

    assert output.size == 128, output.shape


def test_idiap_inceptionv2_casia():
    from bob.bio.face.embeddings import InceptionResnetv2_CasiaWebFace

    np.random.seed(10)
    transformer = InceptionResnetv2_CasiaWebFace()
    data = np.random.rand(3, 160, 160).astype("uint8")
    output = transformer.transform(data)
    assert output.size == 128, output.shape


    # Sample Batch
    sample = Sample(data)
    transformer_sample = wrap(["sample"], transformer)
    output = [s.data for s in transformer_sample.transform([sample])][0]

    assert output.size == 128, output.shape


def test_idiap_inceptionv1_msceleb():
    from bob.bio.face.embeddings import InceptionResnetv1_MsCeleb

    np.random.seed(10)
    transformer = InceptionResnetv1_MsCeleb()
    data = np.random.rand(3, 160, 160).astype("uint8")
    output = transformer.transform(data)
    assert output.size == 128, output.shape

    # Sample Batch
    sample = Sample(data)
    transformer_sample = wrap(["sample"], transformer)
    output = [s.data for s in transformer_sample.transform([sample])][0]

    assert output.size == 128, output.shape


def test_idiap_inceptionv1_casia():
    from bob.bio.face.embeddings import InceptionResnetv1_CasiaWebFace

    np.random.seed(10)
    transformer = InceptionResnetv1_CasiaWebFace()
    data = np.random.rand(3, 160, 160).astype("uint8")
    output = transformer.transform(data)
    assert output.size == 128, output.shape

    # Sample Batch
    sample = Sample(data)
    transformer_sample = wrap(["sample"], transformer)
    output = [s.data for s in transformer_sample.transform([sample])][0]

    assert output.size == 128, output.shape


def test_arface_insight_tf():
    import tensorflow as tf
    tf.compat.v1.reset_default_graph()

    from bob.bio.face.embeddings import ArcFace_InsightFaceTF

    np.random.seed(10)
    transformer = ArcFace_InsightFaceTF()
    data = np.random.rand(3, 112, 112).astype("uint8")
    output = transformer.transform(data)
    assert output.size == 512, output.shape

    # Sample Batch
    sample = Sample(data)
    transformer_sample = wrap(["sample"], transformer)
    output = [s.data for s in transformer_sample.transform([sample])][0]
    assert output.size == 512, output.shape
