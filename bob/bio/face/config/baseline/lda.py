from bob.bio.base.pipelines.vanilla_biometrics import (
    Distance,
    VanillaBiometricsPipeline,
    BioAlgorithmLegacy,
)
from bob.bio.face.config.baseline.helpers import crop_80x64
import numpy as np
import bob.bio.face
from sklearn.pipeline import make_pipeline
from bob.pipelines import wrap
import tempfile
from bob.bio.base.transformers import AlgorithmTransformer
from bob.pipelines.transformers import SampleLinearize
import os

#### SOLVING IF THERE'S ANY DATABASE INFORMATION
if "database" in locals():
    annotation_type = database.annotation_type
    fixed_positions = database.fixed_positions
else:
    annotation_type = None
    fixed_positions = None


####### SOLVING THE FACE CROPPER TO BE USED ##########

# Cropping
face_cropper, transform_extra_arguments = crop_80x64(
    annotation_type, fixed_positions, color_channel="gray"
)

preprocessor = bob.bio.face.preprocessor.TanTriggs(
    face_cropper=face_cropper, dtype=np.float64
)


#### FEATURE EXTRACTOR ######

tempdir = tempfile.TemporaryDirectory()
lda = bob.bio.base.algorithm.LDA(use_pinv=True, pca_subspace_dimension=0.90)

lda_transformer = AlgorithmTransformer(
    lda, projector_file=os.path.join(tempdir.name, "Projector.hdf5")
)


transformer = make_pipeline(
    wrap(
        ["sample"], preprocessor, transform_extra_arguments=transform_extra_arguments,
    ),
    SampleLinearize(),
    wrap(["sample"], lda_transformer),
)


### BIOMETRIC ALGORITHM

algorithm = BioAlgorithmLegacy(
    lda,
    base_dir=tempdir.name,
    projector_file=os.path.join(tempdir.name, "Projector.hdf5"),
)

pipeline = VanillaBiometricsPipeline(transformer, algorithm)
