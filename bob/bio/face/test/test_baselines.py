from bob.extension.config import load
import pkg_resources
import numpy as np
from bob.pipelines import Sample, SampleSet, DelayedSample
from bob.bio.base import load_resource
from bob.bio.base.pipelines.vanilla_biometrics import (
    checkpoint_vanilla_biometrics,
    dask_vanilla_biometrics,
)
import tempfile
import os
import bob.io.base
import functools
import copy
from nose.plugins.attrib import attr

from bob.bio.base.test.utils import is_library_available

images = dict()
images["bioref"] = (
    pkg_resources.resource_filename("bob.bio.face.test", "data/testimage.jpg"),
    {"reye": (131, 176), "leye": (222, 170)},
)
images["probe"] = (
    pkg_resources.resource_filename("bob.bio.face.test", "data/ada.png"),
    {"reye": (440, 207), "leye": (546, 207)},
)


def get_fake_sample_set(face_size=(160, 160), purpose="bioref"):

    data = images[purpose][0]
    annotations = images[purpose][1]
    key = "1" if purpose == "bioref" else "2"

    return [
        SampleSet(
            [
                DelayedSample(
                    load=functools.partial(bob.io.base.load, data),
                    key=key,
                    annotations=annotations,
                )
            ],
            key=key,
            reference_id=key,
            references=["1"],
        )
    ]


def get_fake_samples_for_training():

    data = np.random.rand(10, 3, 400, 400)
    annotations = {"reye": (131, 176), "leye": (222, 170)}

    return [
        Sample(x, key=str(i), reference_id=str(i), annotations=annotations)
        for i, x in enumerate(data)
    ]


def run_baseline(baseline, samples_for_training=[]):
    biometric_references = get_fake_sample_set(purpose="bioref")
    probes = get_fake_sample_set(purpose="probe")

    # Regular pipeline
    pipeline = load_resource(baseline, "pipeline")
    scores = pipeline(samples_for_training, biometric_references, probes)
    assert len(scores) == 1
    assert len(scores[0]) == 1

    # CHECKPOINTING
    with tempfile.TemporaryDirectory() as d:

        cpy = copy.deepcopy(pipeline)
        checkpoint_pipeline = checkpoint_vanilla_biometrics(cpy, base_dir=d)

        checkpoint_scores = checkpoint_pipeline([], biometric_references, probes)
        assert len(checkpoint_scores) == 1
        assert len(checkpoint_scores[0]) == 1
        assert np.isclose(scores[0][0].data, checkpoint_scores[0][0].data)

        dirs = os.listdir(d)
        assert "biometric_references" in dirs
        assert "samplewrapper-1" in dirs
        assert "samplewrapper-2" in dirs
        assert "scores" in dirs

    # DASK

    with tempfile.TemporaryDirectory() as d:
        cpy = copy.deepcopy(pipeline)
        dask_pipeline = dask_vanilla_biometrics(
            checkpoint_vanilla_biometrics(cpy, base_dir=d)
        )
        dask_scores = dask_pipeline([], biometric_references, probes)
        dask_scores = dask_scores.compute(scheduler="single-threaded")
        assert len(dask_scores) == 1
        assert len(dask_scores[0]) == 1
        assert np.isclose(scores[0][0].data, dask_scores[0][0].data)

        dirs = os.listdir(d)
        assert "biometric_references" in dirs
        assert "samplewrapper-1" in dirs
        assert "samplewrapper-2" in dirs
        assert "scores" in dirs


@attr('slow')
@is_library_available("tensorflow")
def test_facenet_baseline():
    run_baseline("facenet-sanderberg")


@attr('slow')
@is_library_available("tensorflow")
def test_inception_resnetv2_msceleb():
    run_baseline("inception-resnetv2-msceleb")


@attr('slow')
@is_library_available("tensorflow")
def test_inception_resnetv2_casiawebface():
    run_baseline("inception-resnetv2-casiawebface")


@attr('slow')
@is_library_available("tensorflow")
def test_inception_resnetv1_msceleb():
    run_baseline("inception-resnetv1-msceleb")


@attr('slow')
@is_library_available("tensorflow")
def test_inception_resnetv1_casiawebface():
    run_baseline("inception-resnetv1-casiawebface")


@attr('slow')
@is_library_available("mxnet")
def test_arcface_insightface():
    run_baseline("arcface-insightface")


def test_gabor_graph():
    run_baseline("gabor_graph")


# def test_lda():
#    run_baseline("lda", get_fake_samples_for_training())
