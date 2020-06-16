#!/usr/bin/env python

from bob.bio.base.pipelines.vanilla_biometrics import DatabaseConnector
from bob.extension import rc
from bob.bio.face.database import ARFaceBioDatabase

arface_directory = rc["bob.db.arface.directory"]

database = DatabaseConnector(
    ARFaceBioDatabase(
        original_directory=arface_directory, original_extension=".png", protocol="all"
    )
)
