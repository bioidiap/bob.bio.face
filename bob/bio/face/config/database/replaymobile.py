#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Yannick Dayer <yannick.dayer@idiap.ch>

"""
    Replay-mobile CSV database interface instantiation
"""

from bob.bio.face.database.replaymobile import ReplayMobileBioDatabase
from bob.extension import rc
import bob.core

logger = bob.core.log.setup("bob.bio.face")

if 'protocol' not in locals():
    logger.info("protocol not specified, using default: 'grandtest'")
    protocol = "grandtest"

logger.debug(f"Instantiation of ReplayMobile bio database with protocol '{protocol}'")
database = ReplayMobileBioDatabase(
    protocol_name=protocol,
    protocol_definition_path=rc.get("bob_data_folder", "~/bob_data/"), # TODO upload the csv files and remove this path.
)
