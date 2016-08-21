#!/usr/bin/env python

from bob.bio.db import ARFaceBioDatabase

arface_directory = "[YOUR_ARFACE_DIRECTORY]"

database = ARFaceBioDatabase(
    original_directory=arface_directory,
    original_extension=".png",
    protocol='all'
)
