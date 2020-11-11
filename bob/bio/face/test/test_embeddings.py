import bob.bio.face
import bob.io.base
import numpy as np
from bob.pipelines import Sample, wrap
import pkg_resources


def test_idiap_inceptionv2_msceleb():
    from bob.bio.face.embeddings import InceptionResnetv2_MsCeleb_CenterLoss_2018

    reference = bob.io.base.load(
        pkg_resources.resource_filename(
            "bob.bio.face.test", "data/inception_resnet_v2_rgb.hdf5"
        )
    )
    np.random.seed(10)
    transformer = InceptionResnetv2_MsCeleb_CenterLoss_2018()
    data = (np.random.rand(3, 160, 160) * 255).astype("uint8")
    output = transformer.transform([data])[0]
    assert output.size == 128, output.shape

    # Sample Batch
    sample = Sample(data)
    transformer_sample = wrap(["sample"], transformer)
    output = [s.data for s in transformer_sample.transform([sample])][0]

    assert np.allclose(output, reference)
    assert output.size == 128, output.shape


def test_idiap_inceptionv2_casia():
    from bob.bio.face.embeddings import InceptionResnetv2_Casia_CenterLoss_2018

    np.random.seed(10)
    transformer = InceptionResnetv2_Casia_CenterLoss_2018()
    data = np.random.rand(3, 160, 160).astype("uint8")
    output = transformer.transform([data])[0]
    assert output.size == 128, output.shape

    # Sample Batch
    sample = Sample(data)
    transformer_sample = wrap(["sample"], transformer)
    output = [s.data for s in transformer_sample.transform([sample])][0]

    assert output.size == 128, output.shape


def test_idiap_inceptionv1_msceleb():
    from bob.bio.face.embeddings import InceptionResnetv1_MsCeleb_CenterLoss_2018

    np.random.seed(10)
    transformer = InceptionResnetv1_MsCeleb_CenterLoss_2018()
    data = np.random.rand(3, 160, 160).astype("uint8")
    output = transformer.transform([data])[0]
    assert output.size == 128, output.shape

    # Sample Batch
    sample = Sample(data)
    transformer_sample = wrap(["sample"], transformer)
    output = [s.data for s in transformer_sample.transform([sample])][0]

    assert output.size == 128, output.shape


def test_idiap_inceptionv1_casia():
    from bob.bio.face.embeddings import InceptionResnetv1_Casia_CenterLoss_2018

    np.random.seed(10)
    transformer = InceptionResnetv1_Casia_CenterLoss_2018()
    data = np.random.rand(3, 160, 160).astype("uint8")
    output = transformer.transform([data])[0]
    assert output.size == 128, output.shape

    # Sample Batch
    sample = Sample(data)
    transformer_sample = wrap(["sample"], transformer)
    output = [s.data for s in transformer_sample.transform([sample])][0]

    assert output.size == 128, output.shape

def test_facenet_sanderberg():
    from bob.bio.face.embeddings import FaceNetSanderberg_20170512_110547

    np.random.seed(10)
    transformer = FaceNetSanderberg_20170512_110547()
    data = np.random.rand(3, 160, 160).astype("uint8")
    output = transformer.transform([data])[0]
    assert output.size == 128, output.shape

    # Sample Batch
    sample = Sample(data)
    transformer_sample = wrap(["sample"], transformer)
    output = [s.data for s in transformer_sample.transform([sample])][0]
    assert output.size == 128, output.shape


def test_arcface_insight_face():
    from bob.bio.face.embeddings import ArcFaceInsightFace

    transformer = ArcFaceInsightFace()
    data = np.random.rand(3, 112, 112)*255
    data = data.astype("uint8")
    output = transformer.transform([data])
    assert output.size == 512, output.shape
    
    # Sample Batch
    sample = Sample(data)
    transformer_sample = wrap(["sample"], transformer)
    output = [s.data for s in transformer_sample.transform([sample])][0]
    assert output.size == 512, output.shape


"""
def test_arface_insight_tf():
    import tensorflow as tf

    tf.compat.v1.reset_default_graph()

    from bob.bio.face.embeddings import ArcFace_InsightFaceTF

    np.random.seed(10)
    transformer = ArcFace_InsightFaceTF()
    data = np.random.rand(3, 112, 112).astype("uint8")
    output = transformer.transform([data])[0]
    assert output.size == 512, output.shape

    # Sample Batch
    sample = Sample(data)
    transformer_sample = wrap(["sample"], transformer)
    output = [s.data for s in transformer_sample.transform([sample])][0]
    assert output.size == 512, output.shape
"""