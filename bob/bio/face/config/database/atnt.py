#!/usr/bin/env python

from bob.bio.db import AtntBioDatabase

atnt_directory = "[YOUR_CAS-PEAL_DIRECTORY]"

database = AtntBioDatabase(
    original_directory=atnt_directory,
)
