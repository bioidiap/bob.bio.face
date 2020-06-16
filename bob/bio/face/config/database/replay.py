#!/usr/bin/env python

from bob.bio.face.database import ReplayBioDatabase
from bob.bio.base.pipelines.vanilla_biometrics import DatabaseConnector
from bob.extension import rc


replay_attack_directory = rc["bob.db.replay.directory"]

replay_licit = DatabaseConnector(
    ReplayBioDatabase(
        original_directory=replay_attack_directory,
        original_extension=".mov",
        protocol="grandtest-licit",
    )
)

replay_spoof = DatabaseConnector(
    ReplayBioDatabase(
        original_directory=replay_attack_directory,
        original_extension=".mov",
        protocol="grandtest-spoof",
    )
)
