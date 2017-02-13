#!/usr/bin/env python

from bob.bio.face.database import MsuMfsdModBioDatabase

msu_mfsd_mod_directory = "[YOUR_MSU_MFSD_MOD_DIRECTORY]"

msu_mfsd_mod_licit = MsuMfsdModBioDatabase(
    original_directory=msu_mfsd_mod_directory,
    original_extension=".mov",
    protocol='grandtest-licit',
)

msu_mfsd_mod_spoof = MsuMfsdModBioDatabase(
    original_directory=msu_mfsd_mod_directory,
    original_extension=".mov",
    protocol='grandtest-spoof',
)
