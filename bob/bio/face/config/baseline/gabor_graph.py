from bob.bio.base.pipelines.vanilla_biometrics import (
    Distance,
    VanillaBiometricsPipeline,
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
import tempfile
import os

import logging

logger = logging.getLogger(__name__)

#### SOLVING IF THERE'S ANY DATABASE INFORMATION
annotation_type, fixed_positions, memory_demanding = lookup_config_from_database(
    locals().get("database")
)


def get_pipeline(face_cropper, transform_extra_arguments):
    preprocessor = bob.bio.face.preprocessor.INormLBP(
        face_cropper=face_cropper, dtype=np.float64
    )

    #### FEATURE EXTRACTOR ######

    # legacy objects needs to be wrapped with legacy transformers
    from bob.bio.base.transformers import ExtractorTransformer

    gabor_graph = ExtractorTransformer(
        bob.bio.face.extractor.GridGraph(
            # Gabor parameters
            gabor_sigma=math.sqrt(2.0) * math.pi,
            # what kind of information to extract
            normalize_gabor_jets=True,
            # setup of the fixed grid
            node_distance=(8, 8),
        )
    )

    transformer = make_pipeline(
        wrap(
            ["sample"],
            preprocessor,
            transform_extra_arguments=transform_extra_arguments,
        ),
        wrap(["sample"], gabor_graph),
    )

    gabor_jet = bob.bio.face.algorithm.GaborJet(
        gabor_jet_similarity_type="PhaseDiffPlusCanberra",
        multiple_feature_scoring="max_jet",
        gabor_sigma=math.sqrt(2.0) * math.pi,
    )

    # Set default temporary directory
    tempdir = bob.bio.base.pipelines.vanilla_biometrics.legacy.get_temp_directory(
        "gabor_graph"
    )

    algorithm = BioAlgorithmLegacy(gabor_jet, base_dir=tempdir)
    return VanillaBiometricsPipeline(transformer, algorithm)


def load(annotation_type, fixed_positions=None):
    ####### SOLVING THE FACE CROPPER TO BE USED ##########

    # Define cropped positions
    CROPPED_IMAGE_HEIGHT = 80
    CROPPED_IMAGE_WIDTH = CROPPED_IMAGE_HEIGHT * 4 // 5
    cropped_image_size = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH)
    cropped_positions = legacy_default_cropping(cropped_image_size, annotation_type)

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
