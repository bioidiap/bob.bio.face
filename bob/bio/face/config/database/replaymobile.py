#!/usr/bin/env python

from bob.bio.face.database import ReplayMobileBioDatabase

replay_mobile_directory = "[YOUR_REPLAY_MOBILE_DIRECTORY]"

replay_licit = ReplayMobileBioDatabase(
    original_directory=replay_mobile_directory,
    original_extension=".mov",
    protocol='grandtest-licit',
)

replay_spoof = ReplayMobileBioDatabase(
    original_directory=replay_mobile_directory,
    original_extension=".mov",
    protocol='grandtest-spoof',
)
