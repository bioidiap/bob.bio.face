#!/usr/bin/env python

import bob.db.multipie
import bob.bio.base

multipie_image_directory = "[YOUR_MULTI-PIE_IMAGE_DIRECTORY]"
multipie_annotation_directory = "[YOUR_MULTI-PIE_ANNOTATION_DIRECTORY]"

database = bob.bio.base.database.DatabaseBobZT(
    database = bob.db.multipie.Database(
        original_directory = multipie_image_directory,
        annotation_directory = multipie_annotation_directory
    ),
    name = "multipie",
    protocol = 'U',
    training_depends_on_protocol = True
)
