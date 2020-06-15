from bob.bio.base.pipelines.vanilla_biometrics import Distance, VanillaBiometricsPipeline, BioAlgorithmLegacy
from bob.bio.face.helpers import face_crop_solver
import math
import numpy as np
import bob.bio.face
from sklearn.pipeline import make_pipeline
from bob.pipelines import wrap
import tempfile

#### SOLVING IF THERE'S ANY DATABASE INFORMATION
if "database" in locals():
    annotation_type = database.annotation_type
    fixed_positions = database.fixed_positions
else:
    annotation_type = None
    fixed_positions = None


####### SOLVING THE FACE CROPPER TO BE USED ##########

# Cropping
CROPPED_IMAGE_HEIGHT = 80
CROPPED_IMAGE_WIDTH = CROPPED_IMAGE_HEIGHT * 4 // 5

# eye positions for frontal images
RIGHT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 - 1)
LEFT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 * 3)

cropped_image_size = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH)
color_channel = "gray"


if annotation_type == "bounding-box":
    transform_extra_arguments = (("annotations", "annotations"),)
    TOP_LEFT_POS = (0, 0)
    BOTTOM_RIGHT_POS = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH)

    # Detects the face and crops it without eye detection
    face_cropper = face_crop_solver(
        cropped_image_size,
        color_channel=color_channel,
        cropped_positions={"topleft": TOP_LEFT_POS, "bottomright": BOTTOM_RIGHT_POS},
        fixed_positions=fixed_positions,
    )

elif annotation_type == "eyes-center":
    transform_extra_arguments = (("annotations", "annotations"),)
    # eye positions for frontal images

    # Detects the face and crops it without eye detection
    face_cropper = face_crop_solver(
        cropped_image_size,
        color_channel=color_channel,
        cropped_positions={"leye": LEFT_EYE_POS, "reye": RIGHT_EYE_POS},
        fixed_positions=fixed_positions,
    )

else:
    transform_extra_arguments = None
    # DEFAULT TO FACE SIMPLE RESIZE
    face_cropper = face_crop_solver(cropped_image_size)

preprocessor = bob.bio.face.preprocessor.INormLBP(
  face_cropper = face_cropper,
  dtype = np.float64
)



#### FEATURE EXTRACTOR ######

gabor_graph = bob.bio.face.extractor.GridGraph(
    # Gabor parameters
    gabor_sigma=math.sqrt(2.0) * math.pi,
    # what kind of information to extract
    normalize_gabor_jets=True,
    # setup of the fixed grid
    node_distance=(8, 8),
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

tempdir = tempfile.TemporaryDirectory()
algorithm = BioAlgorithmLegacy(gabor_jet, base_dir=tempdir.name)

pipeline = VanillaBiometricsPipeline(
    transformer,
    algorithm
)
