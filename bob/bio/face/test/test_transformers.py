import numpy as np
from bob.pipelines import Sample
from bob.bio.base import load_resource
from bob.bio.base.test.utils import is_library_available
import pytest


def get_fake_sample(face_size=(160, 160), eyes={"leye": (46, 107), "reye": (46, 53)}):
    np.random.seed(10)
    data = np.random.rand(3, 400, 400)
    annotations = {"leye": (115, 267), "reye": (115, 132)}
    return Sample(data, key="1", annotations=annotations)


def test_lgbphs():
    transformer = load_resource("lgbphs", "transformer")

    fake_sample = get_fake_sample()
    transformed_sample = transformer.transform([fake_sample])[0]
    transformed_data = transformed_sample.data
    assert transformed_sample.data.shape == (2, 44014)
