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
import numpy as np
import bob.bio.face
from sklearn.pipeline import make_pipeline
from bob.pipelines import wrap
import tempfile
from bob.bio.base.transformers import AlgorithmTransformer
from bob.pipelines.transformers import SampleLinearize
import os

import logging

logger = logging.getLogger(__name__)

#### SOLVING IF THERE'S ANY DATABASE INFORMATION
annotation_type, fixed_positions, memory_demanding = lookup_config_from_database(
    locals().get("database")
)


####### SOLVING THE FACE CROPPER TO BE USED ##########
def load(annotation_type, fixed_positions=None):

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

    preprocessor = bob.bio.face.preprocessor.TanTriggs(
        face_cropper=face_cropper, dtype=np.float64
    )

    #### FEATURE EXTRACTOR ######

    # Set default temporary directory
    tempdir = bob.bio.base.pipelines.vanilla_biometrics.legacy.get_temp_directory("lda")

    lda = bob.bio.base.algorithm.LDA(use_pinv=True, pca_subspace_dimension=0.90)

    lda_transformer = AlgorithmTransformer(
        lda, projector_file=os.path.join(tempdir, "Projector.hdf5")
    )

    transformer = make_pipeline(
        wrap(
            ["sample"],
            preprocessor,
            transform_extra_arguments=transform_extra_arguments,
        ),
        SampleLinearize(),
        wrap(["sample"], lda_transformer),
    )

    ### BIOMETRIC ALGORITHM

    algorithm = BioAlgorithmLegacy(
        lda, base_dir=tempdir, projector_file=os.path.join(tempdir, "Projector.hdf5"),
    )

    return VanillaBiometricsPipeline(transformer, algorithm)


pipeline = load(annotation_type, fixed_positions)

transformer = pipeline.transformer
