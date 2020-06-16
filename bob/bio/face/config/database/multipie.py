#!/usr/bin/env python

from bob.bio.face.database import MultipieBioDatabase
from bob.bio.base.pipelines.vanilla_biometrics import DatabaseConnector
from bob.extension import rc


multipie_image_directory = rc["bob.db.multipie.directory"]
multipie_annotation_directory = rc["bob.db.multipie.annotations"]

database = DatabaseConnector(
    MultipieBioDatabase(
        original_directory=multipie_image_directory,
        annotation_directory=multipie_annotation_directory,
        protocol="U",
        training_depends_on_protocol=True,
    )
)
