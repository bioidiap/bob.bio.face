#!/usr/bin/env python

from bob.bio.face.database import BancaBioDatabase

banca_directory = "[YOUR_BANCA_DIRECTORY]"

database = BancaBioDatabase(
    original_directory=banca_directory,
    original_extension=".ppm",
    protocol='P'
)

