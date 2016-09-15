#!/usr/bin/env python

from bob.bio.face.database import AtntBioDatabase

atnt_directory = "[YOUR_ATNT_DIRECTORY]"

database = AtntBioDatabase(
    original_directory=atnt_directory,
    original_extension=".pgm",
)
