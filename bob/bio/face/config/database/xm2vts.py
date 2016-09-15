#!/usr/bin/env python

from bob.bio.face.database import XM2VTSBioDatabase

xm2vts_directory = "[YOUR_XM2VTS_DIRECTORY]"

database = XM2VTSBioDatabase(
    original_directory=xm2vts_directory,
    protocol='lp1'
)
