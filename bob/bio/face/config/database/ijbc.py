#!/usr/bin/env python

from bob.bio.face.database import IJBCBioDatabase

ijbc_directory = "[YOUR_IJBC_DIRECTORY]"

database = IJBCBioDatabase(
  original_directory=ijbc_directory,
  protocol='1:1'
)
