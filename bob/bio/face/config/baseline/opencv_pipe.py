import bob.bio.base
from bob.bio.face.preprocessor import FaceCrop
from bob.bio.base.transformers.preprocessor import PreprocessorTransformer
from bob.bio.face.extractor import opencv_model
from bob.bio.base.extractor import Extractor
from bob.bio.base.transformers import ExtractorTransformer
from bob.bio.base.algorithm import Distance
from bob.bio.base.pipelines.vanilla_biometrics.legacy import BioAlgorithmLegacy
import scipy.spatial
from bob.bio.base.pipelines.vanilla_biometrics import Distance
from sklearn.pipeline import make_pipeline
from bob.pipelines import wrap
from bob.bio.base.pipelines.vanilla_biometrics import VanillaBiometricsPipeline


memory_demanding = False
if "database" in locals():
    annotation_type = database.annotation_type
    fixed_positions = database.fixed_positions
    memory_demanding = (
        database.memory_demanding if hasattr(database, "memory_demanding") else False
    )

else:
    annotation_type = None
    fixed_positions = None


cropped_positions = {"leye": (98, 144), "reye": (98, 76)}
# Preprocessor
preprocessor_transformer = FaceCrop(
    cropped_image_size=(224, 224),
    cropped_positions={"leye": (98, 144), "reye": (98, 76)},
    color_channel="rgb",
    fixed_positions=fixed_positions,
)

cropped_positions = {"leye": (100, 140), "reye": (100, 95)}
# Preprocessor
preprocessor_transformer = FaceCrop(
    cropped_image_size=(224, 224),
    cropped_positions={"leye": (100, 140), "reye": (100, 95)},
    color_channel="rgb",
    fixed_positions=fixed_positions,
)

transform_extra_arguments = (
    None
    if (cropped_positions is None or fixed_positions is not None)
    else (("annotations", "annotations"),)
)


# Extractor
extractor_transformer = opencv_model()


# Algorithm
algorithm = Distance(
    distance_function=scipy.spatial.distance.cosine, is_distance_function=True
)

## Creation of the pipeline


# Chain the Transformers together
transformer = make_pipeline(
    wrap(
        ["sample"],
        preprocessor_transformer,
        transform_extra_arguments=transform_extra_arguments,
    ),
    wrap(["sample"], extractor_transformer)
    # Add more transformers here if needed
)


# Assemble the Vanilla Biometric pipeline and execute
pipeline = VanillaBiometricsPipeline(transformer, algorithm)
transformer = pipeline.transformer
