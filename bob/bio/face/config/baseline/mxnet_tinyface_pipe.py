import bob.bio.base
from bob.bio.face.preprocessor import FaceCrop
from bob.bio.face.annotator import BobIpTinyface
from bob.bio.face.extractor import MxNetModel

from bob.bio.base.algorithm import Distance
from bob.bio.base.pipelines.vanilla_biometrics.legacy import BioAlgorithmLegacy
import scipy.spatial
from bob.bio.base.pipelines.vanilla_biometrics import Distance
from sklearn.pipeline import make_pipeline
from bob.pipelines import wrap
from bob.bio.base.pipelines.vanilla_biometrics import VanillaBiometricsPipeline


annotator_transformer = BobIpTinyface()

preprocessor_transformer = FaceCrop(cropped_image_size=(112,112), cropped_positions={'leye':(49,72), 'reye':(49,38)}, color_channel='rgb',annotator=annotator_transformer)

extractor_transformer = MxNetModel()


algorithm = Distance(distance_function = scipy.spatial.distance.cosine,is_distance_function = True)

transformer = make_pipeline(
    wrap(["sample"], preprocessor_transformer),
    wrap(["sample"], extractor_transformer)
)

pipeline = VanillaBiometricsPipeline(transformer, algorithm)
transformer = pipeline.transformer