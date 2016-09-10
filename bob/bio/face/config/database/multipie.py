#!/usr/bin/env python

from bob.bio.face.database import MultipieBioDatabase

multipie_image_directory = "[YOUR_MULTI-PIE_IMAGE_DIRECTORY]"
multipie_annotation_directory = "[YOUR_MULTI-PIE_ANNOTATION_DIRECTORY]"

database = MultipieBioDatabase(
    original_directory=multipie_image_directory,
    annotation_directory=multipie_annotation_directory,
    protocol='U',
    training_depends_on_protocol = True
)
