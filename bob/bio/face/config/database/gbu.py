#!/usr/bin/env python

import bob.db.gbu
import bob.bio.base

mbgc_v1_directory = "[YOUR_MBGC-V1_DIRECTORY]"

database = bob.bio.base.database.DatabaseBob(
    database = bob.db.gbu.Database(
        original_directory = mbgc_v1_directory
    ),
    name = "gbu",
    protocol = 'Good',
    models_depend_on_protocol = True,

    all_files_options = { 'subworld': 'x2' },
    extractor_training_options = { 'subworld': 'x2' },
    projector_training_options = { 'subworld': 'x2' },
    enroller_training_options = { 'subworld': 'x2' }
)
