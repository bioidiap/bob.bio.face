#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Sat 20 Aug 15:43:10 CEST 2016

"""
  IJBA database implementation of bob.bio.base.database.BioDatabase interface.
  It is an extension of the database interface, which directly talks to IJBA database, for
  verification experiments (good to use in bob.bio.base framework).
"""

from .database import FaceBioFile
from bob.bio.base.database import BioDatabase, BioFileSet
import os
import bob.io.image


class IJBABioFile(FaceBioFile):
  def __init__(self, f):
    super(IJBABioFile, self).__init__(client_id=f.client_id, path=f.path, file_id=f.id)
    self.f = f

  def load(self, directory, extension=None, add_client_id=False):
    return bob.io.image.load(self.make_path(directory, self.f.extension, add_client_id))

  def make_path(self, directory, extension, add_client_id=True):
    if add_client_id:
      # add client ID to the path, so that a unique path is generated
      # (there might be several identities in each physical file)
      path = "%s-%s%s" % (self.path, self.client_id, extension or '')
    else:
      # do not add the client ID to be able to obtain the original image file
      path = "%s%s" % (self.path,  extension or '')
    return str(os.path.join(directory or '', path))


class IJBABioFileSet(BioFileSet):
  def __init__(self, template):
    super(IJBABioFileSet, self).__init__(file_set_id = template.id, files = [IJBABioFile(f) for f in template.files], path = template.path)


class IJBABioDatabase(BioDatabase):
  """
    IJBA database implementation of :py:class:`bob.bio.base.database.BioDatabase` interface.
    It is an extension of an SQL-based database interface, which directly talks to IJBA database, for
    verification experiments (good to use in bob.bio.base framework).
  """

  def __init__(
      self,
      original_directory=None,
      annotations_directory=None,
      original_extension=None,
      **kwargs
  ):
    # call base class constructors to open a session to the database
    super(IJBABioDatabase, self).__init__(
            name='ijba',
            models_depend_on_protocol=True,
            training_depends_on_protocol=True,
            original_directory=original_directory,
            annotations_directory=annotations_directory,
            original_extension=original_extension,
            **kwargs)

    from bob.db.ijba.query import Database as LowLevelDatabase
    self._db = LowLevelDatabase(original_directory, annotations_directory, original_extension)

  def uses_probe_file_sets(self):
    return True

  def model_ids_with_protocol(self, groups=None, protocol="search_split1", **kwargs):
    return self._db.model_ids(groups=groups, protocol=protocol)

  def objects(self, groups=None, protocol="search_split1", purposes=None, model_ids=None, **kwargs):
    return [IJBABioFile(f) for f in self._db.objects(groups=groups, protocol=protocol, purposes=purposes, model_ids=model_ids, **kwargs)]

  def object_sets(self, groups=None, protocol="search_split1", purposes=None, model_ids=None):
    return [IJBABioFileSet(t) for t in self._db.object_sets(groups=groups, protocol=protocol, purposes=purposes, model_ids=model_ids)]

  def annotations(self, biofile):
    return self._db.annotations(biofile.f)

  def client_id_from_model_id(self, model_id, group='dev'):
    return self._db.get_client_id_from_model_id(model_id)

  def original_file_names(self, files):
    return [f.make_path(self.original_directory, f.f.extension, False) for f in files]
