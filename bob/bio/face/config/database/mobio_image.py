#!/usr/bin/env python

import bob.db.mobio
import bob.bio.base

mobio_image_directory = "[YOUR_MOBIO_IMAGE_DIRECTORY]"
mobio_annotation_directory = "[YOUR_MOBIO_ANNOTATION_DIRECTORY]"

database = bob.bio.base.database.DatabaseBobZT(
    database = bob.db.mobio.Database(
        original_directory = mobio_image_directory,
        original_extension = '.png',
        annotation_directory = mobio_annotation_directory,
    ),
    name = "mobio",
    protocol = 'male',
    models_depend_on_protocol = True
)
