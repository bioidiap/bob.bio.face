#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Sat 20 Aug 15:43:10 CEST 2016

"""
  BANCA database implementation of bob.bio.base.database.ZTDatabase interface.
  It is an extension of an SQL-based database interface, which directly talks to Banca database, for
  verification experiments (good to use in bob.bio.base framework).
"""

from .database import FaceBioFile
from bob.bio.base.database import ZTBioDatabase, BioFile


class BancaBioDatabase(ZTBioDatabase):
    """
    Implements verification API for querying Banca database.
    """

    def __init__(
            self,
            **kwargs
    ):
        # call base class constructors to open a session to the database
        super(BancaBioDatabase, self).__init__(name='banca', **kwargs)

        from bob.db.banca.query import Database as LowLevelDatabase
        self.__db = LowLevelDatabase()

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        return self.__db.model_ids(groups=groups, protocol=protocol)

    def tmodel_ids_with_protocol(self, protocol=None, groups=None, **kwargs):
        return self.__db.tmodel_ids(protocol=protocol, groups=groups, **kwargs)

    def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
        retval = self.__db.objects(groups=groups, protocol=protocol, purposes=purposes, model_ids=model_ids, **kwargs)
        return [FaceBioFile(client_id=f.client_id, path=f.path, file_id=f.id) for f in retval]

    def tobjects(self, groups=None, protocol=None, model_ids=None, **kwargs):
        retval = self.__db.tobjects(groups=groups, protocol=protocol, model_ids=model_ids, **kwargs)
        return [FaceBioFile(client_id=f.client_id, path=f.path, file_id=f.id) for f in retval]

    def zobjects(self, groups=None, protocol=None, **kwargs):
        retval = self.__db.zobjects(groups=groups, protocol=protocol, **kwargs)
        return [FaceBioFile(client_id=f.client_id, path=f.path, file_id=f.id) for f in retval]
