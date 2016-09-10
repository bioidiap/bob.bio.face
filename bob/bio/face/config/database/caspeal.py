#!/usr/bin/env python

from bob.bio.face.database import CaspealBioDatabase

caspeal_directory = "[YOUR_CAS-PEAL_DIRECTORY]"

database = CaspealBioDatabase(
    original_directory=caspeal_directory,
    protocol='lighting'
)

