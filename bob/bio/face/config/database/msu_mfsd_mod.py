#!/usr/bin/env python

from bob.bio.face.database import MsuMfsdModBioDatabase
from bob.bio.base.pipelines.vanilla_biometrics import DatabaseConnector
from bob.extension import rc

msu_mfsd_mod_directory = rc["bob.db.msu_mfsd.directory"]

msu_mfsd_mod_licit = DatabaseConnector(
    MsuMfsdModBioDatabase(
        original_directory=msu_mfsd_mod_directory,
        original_extension=".mov",
        protocol="grandtest-licit",
    )
)

msu_mfsd_mod_spoof = DatabaseConnector(
    MsuMfsdModBioDatabase(
        original_directory=msu_mfsd_mod_directory,
        original_extension=".mov",
        protocol="grandtest-spoof",
    )
)
