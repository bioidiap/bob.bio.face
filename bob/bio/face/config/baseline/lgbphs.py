from bob.bio.base.pipelines import (
    Distance,
    PipelineSimple,
    BioAlgorithmLegacy,
)
from bob.bio.face.utils import (
    lookup_config_from_database,
    legacy_default_cropping,
    make_cropper,
)
import math
import numpy as np
import bob.bio.face
from sklearn.pipeline import make_pipeline
from bob.pipelines import wrap
import bob.math


#### SOLVING IF THERE'S ANY DATABASE INFORMATION
annotation_type, fixed_positions, memory_demanding = lookup_config_from_database(
    locals().get("database")
)


def get_pipeline(face_cropper, transform_extra_arguments):
    preprocessor = bob.bio.face.preprocessor.TanTriggs(
        face_cropper=face_cropper, dtype=np.float64
    )

    #### FEATURE EXTRACTOR ######

    lgbphs = bob.bio.face.extractor.LGBPHS(
        # block setup
        block_size=8,
        block_overlap=0,
        # Gabor parameters
        gabor_sigma=math.sqrt(2.0) * math.pi,
        # LBP setup (we use the defaults)
        # histogram setup
        sparse_histogram=True,
    )

    transformer = make_pipeline(
        wrap(
            ["sample"],
            preprocessor,
            transform_extra_arguments=transform_extra_arguments,
        ),
        wrap(["sample"], lgbphs),
    )

    ### BIOMETRIC ALGORITHM
    histogram = bob.bio.face.algorithm.Histogram(
        distance_function=bob.math.histogram_intersection, is_distance_function=False
    )

    tempdir = bob.bio.base.pipelines.legacy.get_temp_directory("LGBPHS")
    algorithm = BioAlgorithmLegacy(histogram, base_dir=tempdir)

    return PipelineSimple(transformer, algorithm)


def load(annotation_type, fixed_positions=None):
    # Define cropped positions
    CROPPED_IMAGE_HEIGHT = 80
    CROPPED_IMAGE_WIDTH = CROPPED_IMAGE_HEIGHT * 4 // 5
    cropped_image_size = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH)
    cropped_positions = legacy_default_cropping(cropped_image_size, annotation_type)

    ####### SOLVING THE FACE CROPPER TO BE USED ##########
    # Cropping
    face_cropper, transform_extra_arguments = make_cropper(
        cropped_image_size=cropped_image_size,
        cropped_positions=cropped_positions,
        fixed_positions=fixed_positions,
        color_channel="gray",
        annotator="mtcnn",
    )

    return get_pipeline(face_cropper, transform_extra_arguments)


pipeline = load(annotation_type, fixed_positions)
transformer = pipeline.transformer
