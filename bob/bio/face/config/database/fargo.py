#!/usr/bin/env python

from bob.bio.base.pipelines import DatabaseConnector
from bob.bio.face.database import FargoBioDatabase
from bob.extension import rc

fargo_directory = rc["bob.db.fargo.directory"]

database = DatabaseConnector(
    FargoBioDatabase(
        original_directory=fargo_directory,
        original_extension=".png",
        protocol="mc-rgb",
    )
)
