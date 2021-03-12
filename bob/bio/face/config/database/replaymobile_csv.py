#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Yannick Dayer <yannick.dayer@idiap.ch>

"""
  Replay-mobile CSV database interface instantiation
"""

from bob.bio.face.database.replaymobile_csv import ReplayMobileDatabase


import bob.core

logger = bob.core.log.setup("bob.bio.face")

if 'protocol' not in locals():
    logger.info("protocol not specified, using default: 'grandtest'")
    protocol = "grandtest"

logger.debug(f"Instantiation of ReplayMobile bio database with protocol '{protocol}'")
database = ReplayMobileDatabase(protocol_name=protocol, protocol_definition_path="./csv_datasets/replay-mobile/") # TODO upload the csv files and remove this path.
