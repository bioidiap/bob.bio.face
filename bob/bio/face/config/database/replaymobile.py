#!/usr/bin/env python

from bob.bio.face.database import ReplayMobileBioDatabase
from bob.bio.base.pipelines.vanilla_biometrics import DatabaseConnector
from bob.extension import rc


replay_mobile_directory = rc["bob.db.replay_mobile.directory"]

replaymobile_licit = DatabaseConnector(
    ReplayMobileBioDatabase(
        original_directory=replay_mobile_directory,
        original_extension=".mov",
        protocol="grandtest-licit",
    )
)

replaymobile_spoof = DatabaseConnector(
    ReplayMobileBioDatabase(
        original_directory=replay_mobile_directory,
        original_extension=".mov",
        protocol="grandtest-spoof",
    )
)
