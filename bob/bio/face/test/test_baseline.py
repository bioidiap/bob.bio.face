from bob.extension.config import load
import pkg_resources
import numpy as np
from bob.pipelines import Sample, SampleSet


def get_fake_sample(face_size=(160, 160), eyes={"leye": (46, 107), "reye": (46, 53)}):

    data = np.random.rand(3, 400, 400)
    annotations = {"leye": (115, 267), "reye": (115, 132)}
    return Sample(data, key="1", annotations=annotations)


def test_facenet_baseline():
    config_name = pkg_resources.resource_filename(
        "bob.bio.face", "config/baseline/facenet.py"
    )
    transformer = load([config_name]).transformer

    fake_sample = get_fake_sample()

    transformed_sample = transformer.transform([fake_sample])[0]
    transformed_data = transformed_sample.data
    assert transformed_sample.data.size == 128


def test_inception_resnetv2_msceleb():
    config_name = pkg_resources.resource_filename(
        "bob.bio.face", "config/baseline/inception_resnetv2_msceleb.py"
    )
    transformer = load([config_name]).transformer

    fake_sample = get_fake_sample()

    transformed_sample = transformer.transform([fake_sample])[0]
    transformed_data = transformed_sample.data
    assert transformed_sample.data.size == 128


def test_inception_resnetv2_casiawebface():
    config_name = pkg_resources.resource_filename(
        "bob.bio.face", "config/baseline/inception_resnetv2_casiawebface.py"
    )
    transformer = load([config_name]).transformer

    fake_sample = get_fake_sample()

    transformed_sample = transformer.transform([fake_sample])[0]
    transformed_data = transformed_sample.data
    assert transformed_sample.data.size == 128


def test_inception_resnetv1_msceleb():
    config_name = pkg_resources.resource_filename(
        "bob.bio.face", "config/baseline/inception_resnetv1_msceleb.py"
    )
    transformer = load([config_name]).transformer

    fake_sample = get_fake_sample()

    transformed_sample = transformer.transform([fake_sample])[0]
    transformed_data = transformed_sample.data
    assert transformed_sample.data.size == 128


def test_inception_resnetv1_casiawebface():
    config_name = pkg_resources.resource_filename(
        "bob.bio.face", "config/baseline/inception_resnetv1_casiawebface.py"
    )
    transformer = load([config_name]).transformer

    fake_sample = get_fake_sample()

    transformed_sample = transformer.transform([fake_sample])[0]
    transformed_data = transformed_sample.data
    assert transformed_sample.data.size == 128
