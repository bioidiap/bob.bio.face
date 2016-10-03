#!/usr/bin/env python

from bob.bio.face.database import ReplayBioDatabase

replay_attack_directory = "[YOUR_REPLAY_ATTACK_DIRECTORY]"

replay_licit = ReplayBioDatabase(
    original_directory=replay_attack_directory,
    original_extension=".mov",
    protocol='grandtest-licit',
)

replay_spoof = ReplayBioDatabase(
    original_directory=replay_attack_directory,
    original_extension=".mov",
    protocol='grandtest-spoof',
)
