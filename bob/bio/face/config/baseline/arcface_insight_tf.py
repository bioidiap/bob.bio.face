from bob.bio.face.embeddings import ArcFace_InsightFaceTF
from bob.bio.face.config.baseline.helpers import embedding_transformer_112x112


if "database" in locals():
    annotation_type = database.annotation_type
    fixed_positions = database.fixed_positions
else:
    annotation_type = None
    fixed_positions = None


transformer = embedding_transformer_112x112(ArcFace_InsightFaceTF(), annotation_type, fixed_positions)
