#!/usr/bin/env python

from bob.bio.face.database import FRGCBioDatabase

frgc_directory = "[YOUR_FRGC_DIRECTORY]"

database = FRGCBioDatabase(
    original_directory=frgc_directory,
    protocol='2.0.1',
    models_depend_on_protocol=True
)
