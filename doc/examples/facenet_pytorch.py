import bob.bio.base
import bob.bio.face
import bob.bio.face.embeddings.pytorch
import facenet_pytorch

# define aligned image details
cropped_image_size=(224, 224)
# align and crop face such that eyes will be located at these coordinates in the aligned image
cropped_positions = {"leye": (110, 144), "reye": (110, 80)}


# load dataset (needs to be specified on command line first)
annotation_type, fixed_positions, memory_demanding = bob.bio.face.utils.lookup_config_from_database(
    locals().get("database")
)


# load pre-trained inception resnet model from facenet_pytorch
model = facenet_pytorch.InceptionResnetV1(pretrained="vggface2").eval()

# wrap this model to make it accessible in the pipeline
embedding = bob.bio.face.embeddings.pytorch.PyTorchModel(
    model=model,
    preprocessor=lambda x: (x-127.5)/128, # sic, cf: https://github.com/timesler/facenet-pytorch/blob/555aa4bec20ca3e7c2ead14e7e39d5bbce203e4b/models/mtcnn.py#L509
    device="cpu",
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
