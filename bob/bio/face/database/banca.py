#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Sat 20 Aug 15:43:10 CEST 2016


from .database import FaceBioDatabaseWithAnnotations
from bob.bio.base.database import ZTBioDatabase


class BancaBioDatabase(FaceBioDatabaseWithAnnotations, ZTBioDatabase):
  """
    BANCA database implementation of :py:class:`bob.bio.base.database.ZTBioDatabase` interface.
    It is an extension of an SQL-based database interface, which directly talks to Banca database, for
    verification experiments (good to use in bob.bio.base framework).
  """

  def __init__(self, **kwargs):
    from bob.db.banca.query import Database as LowLevelDatabase
    super(BancaBioDatabase, self).__init__(name='banca', database=LowLevelDatabase(), **kwargs)

  def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
    return self._database.model_ids(groups=groups, protocol=protocol)

  def tmodel_ids_with_protocol(self, protocol=None, groups=None, **kwargs):
    return self._database.tmodel_ids(protocol=protocol, groups=groups, **kwargs)

  def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
    return self._make_bio(self._database.objects(groups=groups, protocol=protocol, purposes=purposes, model_ids=model_ids, **kwargs))

  def tobjects(self, groups=None, protocol=None, model_ids=None, **kwargs):
    return self._make_bio(self._database.tobjects(groups=groups, protocol=protocol, model_ids=model_ids, **kwargs))

  def zobjects(self, groups=None, protocol=None, **kwargs):
    return self._make_bio(self._database.zobjects(groups=groups, protocol=protocol, **kwargs))
