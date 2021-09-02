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
import pytest

from bob.bio.base.test.utils import is_library_available

images = dict()
images["bioref"] = (
    pkg_resources.resource_filename("bob.bio.face.test", "data/testimage.jpg"),
    {"reye": (176, 131), "leye": (170, 222)},
)
images["probe"] = (
    pkg_resources.resource_filename("bob.bio.face.test", "data/ada.png"),
    {"reye": (207, 440), "leye": (207, 546)},
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


def run_baseline(baseline, samples_for_training=[], target_scores=None):
    biometric_references = get_fake_sample_set(purpose="bioref")
    probes = get_fake_sample_set(purpose="probe")

    # Regular pipeline
    pipeline = load_resource(baseline, "pipeline")
    scores = pipeline(samples_for_training, biometric_references, probes)
    return scores
    assert len(scores) == 1
    assert len(scores[0]) == 1

    # CHECKPOINTING
    with tempfile.TemporaryDirectory() as d:

        cpy = copy.deepcopy(pipeline)
        checkpoint_pipeline = checkpoint_vanilla_biometrics(cpy, base_dir=d)

        checkpoint_scores = checkpoint_pipeline([], biometric_references, probes)
        assert len(checkpoint_scores) == 1
        assert len(checkpoint_scores[0]) == 1
        if target_scores is not None:
            assert np.allclose(target_scores, scores[0][0].data, atol=10e-5, rtol=10e-5)

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


@pytest.mark.slow
@is_library_available("tensorflow")
def test_facenet_baseline():
    run_baseline("facenet-sanderberg", target_scores=-0.9220775737526933)


@pytest.mark.slow
@is_library_available("tensorflow")
def test_inception_resnetv2_msceleb():
    run_baseline("inception-resnetv2-msceleb", target_scores=-0.43447269718504244)


@pytest.mark.slow
@is_library_available("tensorflow")
def test_inception_resnetv2_casiawebface():
    run_baseline("inception-resnetv2-casiawebface", target_scores=-0.634583944368043)


@pytest.mark.slow
@is_library_available("tensorflow")
def test_inception_resnetv1_msceleb():
    run_baseline("inception-resnetv1-msceleb", target_scores=-0.44497649298306907)


@pytest.mark.slow
@is_library_available("tensorflow")
def test_inception_resnetv1_casiawebface():
    run_baseline("inception-resnetv1-casiawebface", target_scores=-0.6411599976437636)


@pytest.mark.slow
@is_library_available("mxnet")
def test_arcface_insightface():
    run_baseline("arcface-insightface", target_scores=-0.8541907225411619)


@pytest.mark.slow
@is_library_available("tensorflow")
def test_arcface_resnet50_msceleb_v1():
    run_baseline("resnet50-msceleb-arcface-2021", target_scores=-0.799834989589404)


@pytest.mark.slow
@is_library_available("tensorflow")
def test_iresnet50_msceleb_idiap_20210623():
    run_baseline("iresnet50-msceleb-idiap-20210623", target_scores=-1.0540606303938558)


@pytest.mark.slow
@is_library_available("tensorflow")
def test_iresnet100_msceleb_idiap_20210623():
    run_baseline("iresnet100-msceleb-idiap-20210623", target_scores=-1.0353392904250978)


@pytest.mark.slow
@is_library_available("tensorflow")
def test_arcface_resnet50_vgg2_v1():
    run_baseline("resnet50-vgg2-arcface-2021", target_scores=-0.949471222980922)


@pytest.mark.slow
@is_library_available("tensorflow")
def test_arcface_mobilenet_msceleb():
    run_baseline("mobilenetv2-msceleb-arcface-2021", target_scores=-0.5688437955767786)


@pytest.mark.slow
@is_library_available("tensorflow")
def test_arcface_resnet50_msceleb_20210521():
    run_baseline("resnet50-msceleb-arcface-20210521", target_scores=-0.9628566738931277)


def test_gabor_graph():
    run_baseline("gabor_graph", target_scores=0.4385451147418939)


# def test_lda():
#    run_baseline("lda", get_fake_samples_for_training())


@pytest.mark.slow
@is_library_available("torch")
def test_afffe():
    run_baseline(
        "afffe", target_scores=-0.27480835869298026,
    )


@pytest.mark.slow
@is_library_available("torch")
def test_iresnet34():
    run_baseline(
        "iresnet34", target_scores=-0.8302991105719331,
    )


@pytest.mark.slow
@is_library_available("torch")
def test_iresnet50():
    run_baseline(
        "iresnet50", target_scores=-0.8016123867448196,
    )


@pytest.mark.slow
@is_library_available("torch")
def test_iresnet100():
    run_baseline(
        "iresnet100", target_scores=-0.8541905958816157,
    )


@pytest.mark.slow
@is_library_available("torch")
def test_attentionnet():

    run_baseline(
        "attentionnet", target_scores=-0.8856203334291886,
    )


@pytest.mark.slow
@is_library_available("torch")
def test_resnest():

    run_baseline(
        "resnest", target_scores=-0.8548176067335934,
    )


@pytest.mark.slow
@is_library_available("torch")
def test_mobilefacenet():

    run_baseline(
        "mobilefacenet", target_scores=-0.8398221143605292,
    )


@pytest.mark.slow
@is_library_available("torch")
def test_efficientnet():

    run_baseline(
        "efficientnet", target_scores=-0.7978759562781405,
    )


@pytest.mark.slow
@is_library_available("torch")
def test_tfnas():

    run_baseline(
        "tfnas", target_scores=-0.7823820403380854,
    )


@pytest.mark.slow
@is_library_available("torch")
def test_hrnet():

    run_baseline(
        "hrnet", target_scores=-0.6428357755937835,
    )


@pytest.mark.slow
@is_library_available("torch")
def test_rexnet():

    run_baseline(
        "rexnet", target_scores=-0.7886427683602303,
    )


@pytest.mark.slow
@is_library_available("torch")
def test_ghostnet():

    run_baseline(
        "ghostnet", target_scores=-0.7886787784251782,
    )


@pytest.mark.slow
@is_library_available("cv2")
def test_vgg16_oxford():
    run_baseline("vgg16-oxford", target_scores=-0.4142390683487125)
