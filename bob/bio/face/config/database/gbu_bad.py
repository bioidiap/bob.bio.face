#!/usr/bin/env python


from bob.extension import rc
from bob.bio.face.database import GBUDatabase


mbgc_v1_directory = rc["bob.bio.face.gbu.directory"]

database = GBUDatabase(protocol="Bad")

