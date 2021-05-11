#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Yannick Dayer <yannick.dayer@idiap.ch>

"""Replay-mobile CSV database interface configuration

The Replay-Mobile Database for face spoofing consists of 1030 video clips of
photo and video attack attempts to 40 clients, under different lighting
conditions.

The vulnerability analysis pipeline uses single frames extracted from the
videos to be accepted by most face recognition systems.

Feed this file to ``bob bio pipelines`` as configuration:

    $ bob bio pipelines -v -m -c replaymobile-img inception-resnetv2-msceleb

    $ bob bio pipelines -v -m -c my_config/protocol.py replaymobile-img inception-resnetv2-msceleb
"""

from bob.bio.face.database.replaymobile import ReplayMobileBioDatabase
from bob.core import log
from bob.extension import rc

logger = log.setup(__name__)

default_protocol = "grandtest"

if 'protocol' not in locals():
    logger.info(f"protocol not specified, using default: '{default_protocol}'")
    protocol = default_protocol

dataset_protocol_path = rc.get("bob.db.replaymobile.dataset_protocol_path", None) # TODO default
logger.info(f"Loading protocol from '{dataset_protocol_path}'")

data_path = rc.get("bob.db.replaymobile.directory") # TODO default
logger.info(f"Raw data files will be fetched from '{data_path}'")

data_extension = rc.get("bob.db.replaymobile.extension", ".mov")
logger.info(f"Raw data files have the '{data_extension}' extension")

annotations_path = rc.get("bob.db.replaymobile.annotation_path", None) # TODO default
logger.info(f"Annotations files will be fetched from '{annotations_path}'")

logger.debug(f"Instantiation of ReplayMobile bio database with protocol '{protocol}'")
database = ReplayMobileBioDatabase(
    protocol_definition_path=dataset_protocol_path,
    protocol_name=protocol,
    data_path=data_path,
    data_extension=data_extension,
    annotations_path=annotations_path,
)
