#!/usr/bin/env python

from bob.bio.face.database import FargoBioDatabase

fargo_directory = "[YOUR_FARGO_DIRECTORY]"

database = FargoBioDatabase(
    original_directory=fargo_directory,
    original_extension=".png",
    protocol='mc-rgb'
)
