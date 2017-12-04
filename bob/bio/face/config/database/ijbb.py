#!/usr/bin/env python

from bob.bio.face.database import IJBBBioDatabase

ijbb_directory = "[YOUR_IJBB_DIRECTORY]"

database = IJBBBioDatabase(
  original_directory=ijbb_directory,
  protocol='1:1'
)
