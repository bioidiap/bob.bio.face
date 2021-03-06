import bob.bio.face
import bob.io.base
import numpy as np
from bob.pipelines import Sample, wrap
import pkg_resources
from bob.bio.base.test.utils import is_library_available


@is_library_available("tensorflow")
def test_idiap_inceptionv2_msceleb():
    from bob.bio.face.embeddings.tf2_inception_resnet import (
        InceptionResnetv2_MsCeleb_CenterLoss_2018,
    )

    reference = bob.io.base.load(
        pkg_resources.resource_filename(
            "bob.bio.face.test", "data/inception_resnet_v2_msceleb_rgb.hdf5"
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

    np.testing.assert_allclose(output, reference.flatten(), rtol=1e-5, atol=1e-4)
    assert output.size == 128, output.shape


@is_library_available("tensorflow")
def test_idiap_inceptionv2_msceleb_memory_demanding():
    from bob.bio.face.embeddings.tf2_inception_resnet import (
        InceptionResnetv2_MsCeleb_CenterLoss_2018,
    )

    reference = bob.io.base.load(
        pkg_resources.resource_filename(
            "bob.bio.face.test", "data/inception_resnet_v2_msceleb_rgb.hdf5"
        )
    )
    np.random.seed(10)

    transformer = InceptionResnetv2_MsCeleb_CenterLoss_2018(memory_demanding=True)
    data = (np.random.rand(3, 160, 160) * 255).astype("uint8")
    output = transformer.transform([data])[0]
    assert output.size == 128, output.shape

    # Sample Batch
    sample = Sample(data)
    transformer_sample = wrap(["sample"], transformer)
    output = [s.data for s in transformer_sample.transform([sample])][0]

    np.testing.assert_allclose(output[0], reference.flatten(), rtol=1e-5, atol=1e-4)
    assert output.size == 128, output.shape


@is_library_available("tensorflow")
def test_idiap_inceptionv2_casia():
    from bob.bio.face.embeddings.tf2_inception_resnet import (
        InceptionResnetv2_Casia_CenterLoss_2018,
    )

    reference = bob.io.base.load(
        pkg_resources.resource_filename(
            "bob.bio.face.test", "data/inception_resnet_v2_casia_rgb.hdf5"
        )
    )
    np.random.seed(10)
    transformer = InceptionResnetv2_Casia_CenterLoss_2018()
    data = (np.random.rand(3, 160, 160) * 255).astype("uint8")
    output = transformer.transform([data])[0]
    assert output.size == 128, output.shape

    # Sample Batch
    sample = Sample(data)
    transformer_sample = wrap(["sample"], transformer)
    output = [s.data for s in transformer_sample.transform([sample])][0]

    np.testing.assert_allclose(output, reference.flatten(), rtol=1e-5, atol=1e-4)
    assert output.size == 128, output.shape


@is_library_available("tensorflow")
def test_idiap_inceptionv1_msceleb():
    from bob.bio.face.embeddings.tf2_inception_resnet import (
        InceptionResnetv1_MsCeleb_CenterLoss_2018,
    )

    reference = bob.io.base.load(
        pkg_resources.resource_filename(
            "bob.bio.face.test", "data/inception_resnet_v1_msceleb_rgb.hdf5"
        )
    )
    np.random.seed(10)
    transformer = InceptionResnetv1_MsCeleb_CenterLoss_2018()
    data = (np.random.rand(3, 160, 160) * 255).astype("uint8")
    output = transformer.transform([data])[0]
    assert output.size == 128, output.shape

    # Sample Batch
    sample = Sample(data)
    transformer_sample = wrap(["sample"], transformer)
    output = [s.data for s in transformer_sample.transform([sample])][0]

    np.testing.assert_allclose(output, reference.flatten(), rtol=1e-5, atol=1e-4)
    assert output.size == 128, output.shape


@is_library_available("tensorflow")
def test_idiap_inceptionv1_casia():
    from bob.bio.face.embeddings.tf2_inception_resnet import (
        InceptionResnetv1_Casia_CenterLoss_2018,
    )

    reference = bob.io.base.load(
        pkg_resources.resource_filename(
            "bob.bio.face.test", "data/inception_resnet_v1_casia_rgb.hdf5"
        )
    )
    np.random.seed(10)
    transformer = InceptionResnetv1_Casia_CenterLoss_2018()
    data = (np.random.rand(3, 160, 160) * 255).astype("uint8")
    output = transformer.transform([data])[0]
    assert output.size == 128, output.shape

    # Sample Batch
    sample = Sample(data)
    transformer_sample = wrap(["sample"], transformer)
    output = [s.data for s in transformer_sample.transform([sample])][0]

    np.testing.assert_allclose(output, reference.flatten(), rtol=1e-5, atol=1e-4)
    assert output.size == 128, output.shape


@is_library_available("tensorflow")
def test_facenet_sanderberg():
    from bob.bio.face.embeddings.tf2_inception_resnet import (
        FaceNetSanderberg_20170512_110547,
    )

    reference = bob.io.base.load(
        pkg_resources.resource_filename(
            "bob.bio.face.test", "data/facenet_sandberg_20170512-110547.hdf5"
        )
    )
    np.random.seed(10)
    transformer = FaceNetSanderberg_20170512_110547()
    data = (np.random.rand(3, 160, 160) * 255).astype("uint8")
    output = transformer.transform([data])[0]
    assert output.size == 128, output.shape

    # Sample Batch
    sample = Sample(data)
    transformer_sample = wrap(["sample"], transformer)
    output = [s.data for s in transformer_sample.transform([sample])][0]

    np.testing.assert_allclose(output, reference.flatten(), rtol=1e-5, atol=1e-4)
    assert output.size == 128, output.shape


@is_library_available("mxnet")
def test_arcface_insight_face():
    from bob.bio.face.embeddings.mxnet_models import ArcFaceInsightFace

    transformer = ArcFaceInsightFace()
    data = np.random.rand(3, 112, 112) * 255
    data = data.astype("uint8")
    output = transformer.transform([data])
    assert output.size == 512, output.shape

    # Sample Batch
    sample = Sample(data)
    transformer_sample = wrap(["sample"], transformer)
    output = [s.data for s in transformer_sample.transform([sample])][0]
    assert output.size == 512, output.shape
