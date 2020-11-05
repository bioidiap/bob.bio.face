#!/usr/bin/env python

from bob.bio.base.pipelines.vanilla_biometrics import DatabaseConnector
from bob.extension import rc
from bob.bio.face.database import AtntBioDatabase

atnt_directory = rc["bob.db.atnt.directory"]

database = DatabaseConnector(
    AtntBioDatabase(original_directory=atnt_directory, original_extension=".pgm",),
    annotation_type=None,
)
