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
import tempfile
import os

import logging

logger = logging.getLogger(__name__)

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
    face_cropper, transform_extra_arguments = get_cropper(
        annotation_type, fixed_positions
    )
    return get_pipeline(face_cropper, transform_extra_arguments)


pipeline = load(annotation_type, fixed_positions)

transformer = pipeline.transformer
