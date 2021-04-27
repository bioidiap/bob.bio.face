import bob.bio.base
from bob.bio.face.preprocessor import FaceCrop
from bob.bio.face.extractor import MxNetModel
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


cropped_positions = {"leye": (49, 72), "reye": (49, 38)}

preprocessor_transformer = FaceCrop(
    cropped_image_size=(112, 112),
    cropped_positions={"leye": (49, 72), "reye": (49, 38)},
    color_channel="rgb",
    fixed_positions=fixed_positions,
)

transform_extra_arguments = (
    None
    if (cropped_positions is None or fixed_positions is not None)
    else (("annotations", "annotations"),)
)


extractor_transformer = MxNetModel()

algorithm = Distance(
    distance_function=scipy.spatial.distance.cosine, is_distance_function=True
)


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


