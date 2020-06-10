from bob.bio.face.embeddings import InceptionResnetv1_MsCeleb
from bob.bio.face.config.baseline.helpers import embedding_transformer_160x160


if "database" in locals():
    annotation_type = database.annotation_type
    fixed_positions = database.fixed_positions
else:
    annotation_type = None
    fixed_positions = None


transformer = embedding_transformer_160x160(InceptionResnetv1_MsCeleb(), annotation_type, fixed_positions)
