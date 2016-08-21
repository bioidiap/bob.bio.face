#!/usr/bin/env python

from bob.bio.db import SCFaceBioDatabase

scface_directory = "[YOUR_SC_FACE_DIRECTORY]"

database = SCFaceBioDatabase(
    original_directory=scface_directory,
    protocol='combined'
)
