#!/usr/bin/env python

from bob.bio.face.database import IJBCBioDatabase
from bob.bio.base.pipelines.vanilla_biometrics import DatabaseConnector
from bob.extension import rc


ijbc_directory = rc["bob.db.ijbc.directory"]

ijbc_11 = DatabaseConnector(
    IJBCBioDatabase(original_directory=ijbc_directory, protocol="1:1")
)

ijbc_covariates = DatabaseConnector(
    IJBCBioDatabase(
        original_directory=ijbc_directory, protocol="Covariates"
    )
)
