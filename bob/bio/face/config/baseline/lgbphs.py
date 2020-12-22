from bob.bio.base.pipelines.vanilla_biometrics import (
    Distance,
    VanillaBiometricsPipeline,
    BioAlgorithmLegacy,
)
from bob.bio.face.config.baseline.helpers import crop_80x64
import math
import numpy as np
import bob.bio.face
from sklearn.pipeline import make_pipeline
from bob.pipelines import wrap
import bob.math


#### SOLVING IF THERE'S ANY DATABASE INFORMATION
if "database" in locals():
    annotation_type = database.annotation_type
    fixed_positions = database.fixed_positions
else:
    annotation_type = None
    fixed_positions = None


def get_cropper(annotation_type, fixed_positions=None):
    # Cropping
    face_cropper, transform_extra_arguments = crop_80x64(
        annotation_type, fixed_positions, color_channel="gray"
    )
    return face_cropper, transform_extra_arguments


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

    tempdir = bob.bio.base.pipelines.vanilla_biometrics.legacy.get_temp_directory(
        "LGBPHS"
    )
    algorithm = BioAlgorithmLegacy(histogram, base_dir=tempdir)

    return VanillaBiometricsPipeline(transformer, algorithm)


def load(annotation_type, fixed_positions=None):
    ####### SOLVING THE FACE CROPPER TO BE USED ##########
    face_cropper, transform_extra_arguments = get_cropper(
        annotation_type, fixed_positions
    )
    return get_pipeline(face_cropper, transform_extra_arguments)


pipeline = load(annotation_type, fixed_positions)
transformer = pipeline.transformer
