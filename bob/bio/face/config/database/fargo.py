#!/usr/bin/env python

from bob.bio.face.database import FargoBioDatabase

fargo_directory = "/Users/guillaumeheusch/idiap/data/fargo/"

database = FargoBioDatabase(
    original_directory=fargo_directory,
    original_extension=".png",
)
