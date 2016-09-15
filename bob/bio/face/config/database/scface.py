#!/usr/bin/env python

from bob.bio.face.database import SCFaceBioDatabase

scface_directory = "[YOUR_SC_FACE_DIRECTORY]"

database = SCFaceBioDatabase(
    original_directory=scface_directory,
    protocol='combined'
)
