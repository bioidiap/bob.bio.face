#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Wed 20 July 14:43:22 CEST 2016

"""
  Verification API for bob.db.voxforge
"""

from bob.bio.base.database.file import BioFile


class FaceBioFile(BioFile):
    def __init__(self, client_id, path, file_id, **kwargs):
        """
        Initializes this File object with an File equivalent for
        VoxForge database.
        """
        super(FaceBioFile, self).__init__(
            client_id=client_id, path=path, file_id=file_id, **kwargs
        )
