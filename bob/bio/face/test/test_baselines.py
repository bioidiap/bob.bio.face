from bob.extension.config import load
import pkg_resources
import numpy as np
from bob.pipelines import Sample, SampleSet
from bob.bio.base import load_resource


def get_fake_sample_set(
    face_size=(160, 160), eyes={"leye": (46, 107), "reye": (46, 53)}
):

    data = np.random.rand(3, 400, 400)
    annotations = {"leye": (115, 267), "reye": (115, 132)}
    return [
        SampleSet(
            [Sample(data, key="1", annotations=annotations)],
            key="1",
            subject="1",
            references=["1"],
        )
    ]


def test_facenet_baseline():

    biometric_references = get_fake_sample_set()
    probes = get_fake_sample_set()
    
    # Regular pipeline
    pipeline = load_resource("facenet_sanderberg", "baseline")
    scores = pipeline([], biometric_references, probes)
    assert len(scores)==1
    assert len(scores[0])==1


    # Regular with 

    # fake_sample = get_fake_sample()

    # transformed_sample = transformer.transform([fake_sample])[0]
    # transformed_data = transformed_sample.data
    # assert transformed_sample.data.size == 128


def test_inception_resnetv2_msceleb():
    transformer = load_resource("inception_resnetv2_msceleb", "baseline")

    fake_sample = get_fake_sample()

    transformed_sample = transformer.transform([fake_sample])[0]
    transformed_data = transformed_sample.data
    assert transformed_sample.data.size == 128


def test_inception_resnetv2_casiawebface():
    transformer = load_resource("inception_resnetv2_casiawebface", "baseline")

    fake_sample = get_fake_sample()

    transformed_sample = transformer.transform([fake_sample])[0]
    transformed_data = transformed_sample.data
    assert transformed_sample.data.size == 128


def test_inception_resnetv1_msceleb():
    transformer = load_resource("inception_resnetv1_msceleb", "baseline")

    fake_sample = get_fake_sample()

    transformed_sample = transformer.transform([fake_sample])[0]
    transformed_data = transformed_sample.data
    assert transformed_sample.data.size == 128


def test_inception_resnetv1_casiawebface():
    transformer = load_resource("inception_resnetv1_casiawebface", "baseline")

    fake_sample = get_fake_sample()

    transformed_sample = transformer.transform([fake_sample])[0]
    transformed_data = transformed_sample.data
    assert transformed_sample.data.size == 128


def test_arcface_insight_tf():
    import tensorflow as tf

    tf.compat.v1.reset_default_graph()
    transformer = load_resource("arcface_insight_tf", "baseline")

    fake_sample = get_fake_sample()

    transformed_sample = transformer.transform([fake_sample])[0]
    transformed_data = transformed_sample.data
    assert transformed_sample.data.size == 512
