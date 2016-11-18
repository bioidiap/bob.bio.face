#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Sat 20 Aug 15:43:10 CEST 2016

"""
  ARFACE database implementation of bob.bio.base.database.ZTDatabase interface.
  It is an extension of an SQL-based database interface, which directly talks to ARFACE database, for
  verification experiments (good to use in bob.bio.base framework).
"""

from .database import FaceBioFileWithAnnotations, FaceBioDatabaseWithAnnotations


class ARFaceBioDatabase(FaceBioDatabaseWithAnnotations):
  """Implements verification API for querying ARface database.
  """

  def __init__(self, **kwargs):
    from bob.db.arface.query import Database as LowLevelDatabase
    super(ARFaceBioDatabase, self).__init__(name='arface', database=LowLevelDatabase(), **kwargs)

  def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
    return self._database.model_ids(groups=groups, protocol=protocol)

  def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
    retval = self._database.objects(groups=groups, protocol=protocol, purposes=purposes, model_ids=model_ids, **kwargs)
    return [FaceBioFileWithAnnotations(client_id=f.client_id, file=f) for f in retval]
