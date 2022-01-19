import bob.bio.base
import bob.bio.face
import numpy

# define aligned image details
cropped_image_size=(112, 112)
# align and crop face such that eyes will be located at these coordinates in the aligned image; additionally provide bounding box information
cropped_positions = {"leye": (20, 88), "reye": (20, 24), "topleft": (-10, 0), "bottomright": (102, 112)}


# load dataset (needs to be specified on command line first)
annotation_type, fixed_positions, memory_demanding = bob.bio.face.utils.lookup_config_from_database(
    locals().get("database")
)

# define our preprocessing
mean = numpy.array([0.485, 0.456, 0.406])
std = numpy.array([0.229, 0.224, 0.225])
def preprocessor(x):
    # get into [0,1] range
    x = x / 255.
    # subtract mean and std
    x = (x - mean) / std

    return x

# our pre-trained model to make it accessible in the pipeline
embedding = bob.bio.face.embeddings.tensorflow.TensorflowTransformer(
    checkpoint_path="models/MyTrainedModel",
    preprocessor=preprocessor,
    memory_demanding=memory_demanding
)

# connect preprocessing and feature extraction into one transformer
transformer = bob.bio.face.utils.embedding_transformer(
    cropped_image_size=cropped_image_size,
    embedding=embedding,
    cropped_positions=cropped_positions,
    fixed_positions=fixed_positions,
    color_channel="rgb",
    annotator=None
)

# define algorithm to compute cosine distances
algorithm = bob.bio.base.pipelines.vanilla_biometrics.Distance()

# connect transformer and algorithm into one pipeline
pipeline = bob.bio.base.pipelines.vanilla_biometrics.VanillaBiometricsPipeline(transformer, algorithm)
