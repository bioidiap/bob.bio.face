#!/usr/bin/env python

from bob.bio.face.database import GBUBioDatabase

mbgc_v1_directory = "[YOUR_MBGC-V1_DIRECTORY]"

database = GBUBioDatabase(
    original_directory=mbgc_v1_directory,
    protocol='Good',
    models_depend_on_protocol=True,

    all_files_options={'subworld': 'x2'},
    extractor_training_options={'subworld': 'x2'},
    projector_training_options={'subworld': 'x2'},
    enroller_training_options={'subworld': 'x2'}
)

