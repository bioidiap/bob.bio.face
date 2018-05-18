#!/usr/bin/env python

from bob.bio.face.database import IJBCBioDatabase

ijbc_directory = "[YOUR_IJBC_DIRECTORY]"

ijbc_11 = IJBCBioDatabase(
  original_directory=ijbc_directory,
  protocol='1:1'
)

ijbc_covariates = IJBCBioDatabase(
  original_directory=ijbc_directory,
  protocol='Covariates'
)
