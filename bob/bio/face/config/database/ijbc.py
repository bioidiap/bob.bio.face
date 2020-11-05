#!/usr/bin/env python

from bob.bio.face.database import IJBCBioDatabase
from bob.bio.base.pipelines.vanilla_biometrics import DatabaseConnector
from bob.extension import rc


ijbc_directory = rc["bob.db.ijbc.directory"]

database = DatabaseConnector(
    IJBCBioDatabase(original_directory=ijbc_directory, protocol="1:1"),
    annotation_type = "eyes-center",
    fixed_positions = None,
)

#ijbc_covariates = DatabaseConnector(
#    IJBCBioDatabase(
#        original_directory=ijbc_directory, protocol="Covariates"
#    )
#)
