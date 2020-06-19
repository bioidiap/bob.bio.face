#!/usr/bin/env python

from bob.bio.base.pipelines.vanilla_biometrics import DatabaseConnector
from bob.extension import rc
from bob.bio.face.database import FargoBioDatabase

fargo_directory = rc["bob.db.fargo.directory"]

database = DatabaseConnector(
    FargoBioDatabase(
        original_directory=fargo_directory, original_extension=".png", protocol="mc-rgb"
    )
)
