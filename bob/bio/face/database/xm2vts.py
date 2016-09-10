#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Sat 20 Aug 15:43:10 CEST 2016

"""
  XM2VTS database implementation of bob.bio.base.database.Database interface.
  It is an extension of an SQL-based database interface, which directly talks to XM2VTS database, for
  verification experiments (good to use in bob.bio.base framework).
"""

from .database import FaceBioFile
from bob.bio.base.database import BioDatabase, BioFile


class XM2VTSBioDatabase(BioDatabase):
    """
    Implements verification API for querying XM2VTS database.
    """

    def __init__(
            self,
            **kwargs
    ):
        # call base class constructors to open a session to the database
        super(XM2VTSBioDatabase, self).__init__(name='xm2vts', **kwargs)

        from bob.db.xm2vts.query import Database as LowLevelDatabase
        self.__db = LowLevelDatabase()

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        return self.__db.model_ids(groups=groups, protocol=protocol)

    def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
        retval = self.__db.objects(groups=groups, protocol=protocol, purposes=purposes, model_ids=model_ids, **kwargs)
        return [FaceBioFile(BioFile(client_id=f.client_id, path=f.path, file_id=f.id)) for f in retval]
