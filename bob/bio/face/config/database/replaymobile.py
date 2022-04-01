#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Yannick Dayer <yannick.dayer@idiap.ch>

"""Replay-mobile CSV database interface configuration

The Replay-Mobile Database for face spoofing consists of video clips of
photo and video attack attempts under different lighting conditions.

The vulnerability analysis pipeline uses single frames extracted from the
videos to be accepted by most face recognition systems.

Feed this file (defined as resource: ``replaymobile-img``) to ``bob bio pipelines`` as
configuration:

    $ bob bio pipeline simple -v --write-metadata-scores replaymobile-img inception-resnetv2-msceleb

    $ bob bio pipeline simple -v --write-metadata-scores my_config/protocol.py replaymobile-img inception-resnetv2-msceleb
"""

from bob.bio.face.database.replaymobile import ReplayMobileBioDatabase

default_protocol = "grandtest"

if "protocol" not in locals():
    protocol = default_protocol

database = ReplayMobileBioDatabase(
    protocol=protocol,
)
