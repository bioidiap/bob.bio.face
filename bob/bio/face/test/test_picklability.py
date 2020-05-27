import bob.bio.face
import bob.bio.base
import numpy as np
import pickle
import nose

import bob.ip.base
import bob.ip.flandmark
import bob.ip.gabor

def assert_picklable_with_exceptions(obj):
    """Test if an object is picklable or not."""

    # some bob C bind objects doesn't have __eq__ properly implemented
    # and we'll not implement it
    # therefore, exception list has been done, so we can skip
    # those objects

    exception_list = [
        bob.ip.base.LBP,
        bob.bio.face.preprocessor.FaceCrop,
        bob.ip.facedetect.detector.sampler.Sampler,
        bob.ip.facedetect.detector.cascade.Cascade,
        bob.ip.flandmark.Flandmark,
        bob.ip.gabor.Similarity,
        bob.ip.gabor.Transform,
        bob.ip.gabor.Graph,
    ]

    string = pickle.dumps(obj)
    new_obj = pickle.loads(string)
    obj = obj.__dict__
    new_obj = new_obj.__dict__
    assert len(obj) == len(new_obj)
    nose.tools.assert_equal(sorted(list(obj.keys())), sorted(list(new_obj.keys())))
    for k, v in obj.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_equal(v, new_obj[k])
        else:
            if type(v) not in exception_list:
                nose.tools.assert_equal(v, new_obj[k])
    return True


### Preprocessors


def test_face_crop():
    CROPPED_IMAGE_HEIGHT = 64
    CROPPED_IMAGE_WIDTH = 64
    RIGHT_EYE_POS = (16.0, 15.5)
    LEFT_EYE_POS = (16.0, 48.0)
    cropper = bob.bio.face.preprocessor.FaceCrop(
        cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
        cropped_positions={"leye": LEFT_EYE_POS, "reye": RIGHT_EYE_POS},
        color_channel="rgb",
        dtype="uint8",
    )
    assert assert_picklable_with_exceptions(cropper)


def test_face_detect():
    face_detect = bob.bio.face.preprocessor.FaceDetect(face_cropper="face-crop-eyes")
    assert_picklable_with_exceptions(face_detect)

    face_detect = bob.bio.face.preprocessor.FaceDetect(
        face_cropper="face-crop-eyes", use_flandmark=True
    )
    assert assert_picklable_with_exceptions(face_detect)


def test_INormLBP():
    face_crop = bob.bio.face.preprocessor.INormLBP(face_cropper="face-crop-eyes")
    assert assert_picklable_with_exceptions(face_crop)


def test_TanTriggs():
    face_crop = bob.bio.face.preprocessor.TanTriggs(face_cropper="face-crop-eyes")
    assert assert_picklable_with_exceptions(face_crop)


def test_SQI():
    face_crop = bob.bio.face.preprocessor.SelfQuotientImage(
        face_cropper="face-crop-eyes"
    )
    assert assert_picklable_with_exceptions(face_crop)


def test_HistogramEqualization():
    face_crop = bob.bio.face.preprocessor.HistogramEqualization(
        face_cropper="face-crop-eyes"
    )
    assert assert_picklable_with_exceptions(face_crop)


### Extractors


def test_DCT():
    extractor = bob.bio.face.extractor.DCTBlocks()
    assert_picklable_with_exceptions(extractor)


def test_GridGraph():
    extractor = bob.bio.face.extractor.GridGraph(node_distance=24)
    #assert assert_picklable_with_exceptions(extractor)

    fake_image = np.arange(64 * 80).reshape(64, 80).astype("float")
    extractor(fake_image)
    assert assert_picklable_with_exceptions(extractor)

    #cropper = bob.bio.base.load_resource(
    #    "face-crop-eyes", "preprocessor", preferred_package="bob.bio.face"
    #)
    #eyes = cropper.cropped_positions
    #extractor = bob.bio.face.extractor.GridGraph(eyes=eyes)
    #assert assert_picklable_with_exceptions(extractor)


def test_LGBPHS():
    import math

    extractor = bob.bio.face.extractor.LGBPHS(
        block_size=8,
        block_overlap=0,
        gabor_directions=4,
        gabor_scales=2,
        gabor_sigma=math.sqrt(2.0) * math.pi,
        sparse_histogram=True,
    )

    assert assert_picklable_with_exceptions(extractor)


## Algorithms


def test_GaborJet():

    algorithm = bob.bio.face.algorithm.GaborJet(
        "PhaseDiffPlusCanberra", multiple_feature_scoring="average_model"
    )
    assert assert_picklable_with_exceptions(algorithm)


def test_Histogram():
    algorithm = bob.bio.face.algorithm.Histogram()
    assert assert_picklable_with_exceptions(algorithm)
