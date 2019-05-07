#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
  FARGO database implementation of bob.bio.base.database.Database interface.
  It is an extension of an SQL-based database interface, which directly talks to FARGO database, for
  verification experiments (good to use in bob.bio.base framework).
"""

import os
import bob.db.base

from .database import FaceBioFile
from bob.bio.base.database import BioDatabase


class FargoBioDatabase(BioDatabase):
    """
    FARGO database implementation of :py:class:`bob.bio.base.database.BioDatabase` interface.
    It is an extension of the database interface, which directly talks to ATNT database, for
    verification experiments (good to use in bob.bio.base framework).
    """

    def __init__(
            self,
            original_directory=None,
            original_extension='.png',
            protocol='mc-rgb',
            **kwargs
    ):
        # call base class constructors to open a session to the database
        super(FargoBioDatabase, self).__init__(
            name='fargo',
            original_directory=original_directory,
            original_extension=original_extension,
            protocol=protocol,
            **kwargs)

        from bob.db.fargo.query import Database as LowLevelDatabase
        self._db = LowLevelDatabase(original_directory, original_extension, protocol=protocol)

    def objects(self, groups=None, purposes=None, protocol=None, model_ids=None, **kwargs):
        retval = self._db.objects(protocol=protocol, groups=groups, purposes=purposes, model_ids=model_ids, **kwargs)
        return [FaceBioFile(client_id=f.client_id, path=f.path, file_id=f.id) for f in retval]
    
    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        return self._db.model_ids(groups=groups, protocol=protocol)

    def annotations(self, file):
      
      if self.annotation_directory is None:
          return None

      annotation_file = os.path.join(self.annotation_directory, file.path + self.annotation_extension)
      return bob.db.base.read_annotation_file(annotation_file, 'eyecenter')
  
    
