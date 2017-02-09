#!/usr/bin/env python

from bob.bio.face.database import MobioBioDatabase

mobio_image_directory = "[YOUR_MOBIO_IMAGE_DIRECTORY]"
mobio_annotation_directory = "[YOUR_MOBIO_ANNOTATION_DIRECTORY]"

mobio_image = MobioBioDatabase(
    original_directory=mobio_image_directory,
    original_extension=".png",
    annotation_directory=mobio_annotation_directory,
    annotation_type="eyecenter",

    protocol='male',
    models_depend_on_protocol=True,
)

mobio_male = MobioBioDatabase(
    original_directory=mobio_image_directory,
    original_extension=".png",
    annotation_directory=mobio_annotation_directory,
    annotation_type="eyecenter",

    protocol='male',
    models_depend_on_protocol=True,

    all_files_options={'gender': 'male'},
    extractor_training_options={'gender': 'male'},
    projector_training_options={'gender': 'male'},
    enroller_training_options={'gender': 'male'},
    z_probe_options={'gender': 'male'}
)


mobio_female = MobioBioDatabase(
    original_directory=mobio_image_directory,
    original_extension=".png",
    annotation_directory=mobio_annotation_directory,
    annotation_type="eyecenter",

    protocol='female',
    models_depend_on_protocol=True,

    all_files_options={'gender': 'female'},
    extractor_training_options={'gender': 'female'},
    projector_training_options={'gender': 'female'},
    enroller_training_options={'gender': 'female'},
    z_probe_options={'gender': 'female'}
)
