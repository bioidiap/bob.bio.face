from bob.bio.base.pipelines.vanilla_biometrics import (
    Distance,
    VanillaBiometricsPipeline,
    BioAlgorithmLegacy,
    temp_directory
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


def load(annotation_type, fixed_positions=None, checkpoints_dir=None):
    ####### SOLVING THE FACE CROPPER TO BE USED ##########

    # Cropping
    face_cropper, transform_extra_arguments = crop_80x64(
        annotation_type, fixed_positions, color_channel="gray"
    )

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
            ["sample"], preprocessor, transform_extra_arguments=transform_extra_arguments,
        ),
        wrap(["sample"], gabor_graph),
    )


    gabor_jet = bob.bio.face.algorithm.GaborJet(
        gabor_jet_similarity_type="PhaseDiffPlusCanberra",
        multiple_feature_scoring="max_jet",
        gabor_sigma=math.sqrt(2.0) * math.pi,
    )

    # Set default temporary directory
    default_temp = os.path.join("/idiap","temp",os.environ["USER"])
    if os.path.exists(default_temp):
        tempdir = os.path.join(default_temp, "bob_bio_base_tmp")
    else:
        # if /idiap/temp/<USER> does not exist, use /tmp/tmpxxxxxxxx
        tempdir = tempfile.TemporaryDirectory().name

    # Replace the default if provided
    if checkpoints_dir is not None:
        try:
            os.makedirs(checkpoints_dir, exist_ok=True)
        except OSError:
            logger.info(
                "Could not create directory '{}'.".format(checkpoints_dir)
                + " Using default ('{}').".format(tempdir)
            )
        else:
            tempdir = checkpoints_dir

    algorithm = BioAlgorithmLegacy(gabor_jet, base_dir=tempdir)
    return VanillaBiometricsPipeline(transformer, algorithm)

try: temp_directory
except NameError:
    logger.info("Temporary directory not defined. Using default.")
    pipeline = load(annotation_type, fixed_positions, None)
else:
    pipeline = load(annotation_type, fixed_positions, temp_directory)

transformer = pipeline.transformer
