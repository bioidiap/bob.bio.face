from bob.extension.config import load
import pkg_resources
import numpy as np
from bob.pipelines import Sample, SampleSet
from bob.bio.base import load_resource
from bob.bio.base.test.utils import is_library_available
from nose.plugins.attrib import attr


def get_fake_sample(face_size=(160, 160), eyes={"leye": (46, 107), "reye": (46, 53)}):
    np.random.seed(10)
    data = np.random.rand(3, 400, 400)
    annotations = {"leye": (115, 267), "reye": (115, 132)}
    return Sample(data, key="1", annotations=annotations)


@attr('slow')
@is_library_available("tensorflow")
def test_facenet_sanderberg():
    transformer = load_resource("facenet-sanderberg", "transformer")

    fake_sample = get_fake_sample()

    transformed_sample = transformer.transform([fake_sample])[0]
    transformed_data = transformed_sample.data
    assert transformed_sample.data.size == 128


@attr('slow')
@is_library_available("tensorflow")
def test_inception_resnetv2_msceleb():
    transformer = load_resource("inception-resnetv2-msceleb", "transformer")

    fake_sample = get_fake_sample()

    transformed_sample = transformer.transform([fake_sample])[0]
    transformed_data = transformed_sample.data
    assert transformed_sample.data.size == 128


@attr('slow')
@is_library_available("tensorflow")
def test_inception_resnetv2_casiawebface():
    transformer = load_resource("inception-resnetv2-casiawebface", "transformer")

    fake_sample = get_fake_sample()

    transformed_sample = transformer.transform([fake_sample])[0]
    transformed_data = transformed_sample.data
    assert transformed_sample.data.size == 128


@attr('slow')
@is_library_available("tensorflow")
def test_inception_resnetv1_msceleb():
    transformer = load_resource("inception-resnetv1-msceleb", "transformer")

    fake_sample = get_fake_sample()

    transformed_sample = transformer.transform([fake_sample])[0]
    transformed_data = transformed_sample.data
    assert transformed_sample.data.size == 128


@attr('slow')
@is_library_available("tensorflow")
def test_inception_resnetv1_casiawebface():
    transformer = load_resource("inception-resnetv1-casiawebface", "transformer")

    fake_sample = get_fake_sample()

    transformed_sample = transformer.transform([fake_sample])[0]
    transformed_data = transformed_sample.data
    assert transformed_sample.data.size == 128


"""
def test_arcface_insight_tf():
    import tensorflow as tf
    tf.compat.v1.reset_default_graph()
    transformer = load_resource("arcface-insight-tf", "transformer")

    fake_sample = get_fake_sample()

    transformed_sample = transformer.transform([fake_sample])[0]
    transformed_data = transformed_sample.data
    assert transformed_sample.data.size == 512
"""


def test_gabor_graph():
    transformer = load_resource("gabor-graph", "transformer")

    fake_sample = get_fake_sample()
    transformed_sample = transformer.transform([fake_sample])[0]
    transformed_data = transformed_sample.data
    assert len(transformed_sample.data) == 80


def test_lgbphs():
    transformer = load_resource("lgbphs", "transformer")

    fake_sample = get_fake_sample()
    transformed_sample = transformer.transform([fake_sample])[0]
    transformed_data = transformed_sample.data
    assert transformed_sample.data.shape == (2, 44014)
