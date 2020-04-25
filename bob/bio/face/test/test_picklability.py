import bob.bio.face
import bob.bio.base
from bob.pipelines.utils import assert_picklable
import numpy

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
    assert assert_picklable(cropper)


def test_face_detect():
    face_detect = bob.bio.face.preprocessor.FaceDetect(face_cropper="face-crop-eyes")
    assert_picklable(face_detect)

    face_detect = bob.bio.face.preprocessor.FaceDetect(
        face_cropper="face-crop-eyes", use_flandmark=True
    )
    assert assert_picklable(face_detect)


def test_INormLBP():
    face_crop = bob.bio.face.preprocessor.INormLBP(face_cropper="face-crop-eyes")
    assert assert_picklable(face_crop)


def test_TanTriggs():
    face_crop = bob.bio.face.preprocessor.TanTriggs(face_cropper="face-crop-eyes")
    assert assert_picklable(face_crop)


def test_SQI():
    face_crop = bob.bio.face.preprocessor.SelfQuotientImage(
        face_cropper="face-crop-eyes"
    )
    assert assert_picklable(face_crop)


def test_HistogramEqualization():
    face_crop = bob.bio.face.preprocessor.HistogramEqualization(
        face_cropper="face-crop-eyes"
    )
    assert assert_picklable(face_crop)


### Extractors


def test_DCT():
    extractor = bob.bio.face.extractor.DCTBlocks()
    assert_picklable(extractor)


def test_GridGraph():    
    extractor = bob.bio.face.extractor.GridGraph(node_distance=24)
    assert assert_picklable(extractor)

    fake_image = numpy.arange(64*80).reshape(64,80).astype("float")
    extractor(fake_image)
    assert assert_picklable(extractor)


    cropper = bob.bio.base.load_resource(
        "face-crop-eyes", "preprocessor", preferred_package="bob.bio.face"
    )
    eyes = cropper.cropped_positions
    extractor = bob.bio.face.extractor.GridGraph(eyes=eyes)
    assert assert_picklable(extractor)





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

    assert assert_picklable(extractor)


## Algorithms


def test_GaborJet():

    algorithm = bob.bio.face.algorithm.GaborJet(
        "PhaseDiffPlusCanberra", multiple_feature_scoring="average_model"
    )
    assert assert_picklable(algorithm)


def test_Histogram():
    algorithm = bob.bio.face.algorithm.Histogram()
    assert assert_picklable(algorithm)
