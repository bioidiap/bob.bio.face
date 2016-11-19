#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Wed 20 July 14:43:22 CEST 2016

from bob.bio.base.database import BioFile, BioDatabase


class FaceBioFile(BioFile):
  def __init__(self, client_id, path, file_id):
    """Initializes this File object with an File equivalent for
    face databases.
    """
    super(FaceBioFile, self).__init__(client_id=client_id, path=path, file_id=file_id)


class FaceBioFileWithAnnotations(FaceBioFile):
  """This class is a wrapper for the :py:class:`FaceBioFile` that stores the given :py:class:`bob.db.base.File` object.
  It can be used inside a :py:class:`FaceBioDatabaseWithAnnotations` to obtain the annotations stored in the low-level database.
  """
  def __init__(self, file, client_id):
    super(FaceBioFileWithAnnotations, self).__init__(client_id = client_id, file_id=file.id, path = file.path)
    self._f = file


class FaceBioDatabaseWithAnnotations(BioDatabase):
  """This class overwrites the default the :py:meth:`bob.bio.base.database.BioDatabase.annotations` function by returning the database that is stored in the low-level database.
  """
  def __init__(self, database, **kwargs):
    super(FaceBioDatabaseWithAnnotations, self).__init__(database=database, **kwargs)
    self._database = database

  def _make_bio(self, files):
    return [FaceBioFileWithAnnotations(client_id=f.client_id, file=f) for f in files]

  def annotations(self, file):
    """annotations(self, file) -> annotations

    Returns the annotations for the given file by querying the database for annotations.

    **Parameters:**

    ``file`` : :py:class:`FaceBioFileWithAnnotations` or derived
      The file to query the annotations for.

    **Returns:**

    ``annotations`` : dict
      The dictionary of annotations, usually containing at least the eye locations as:
      ``{'reye' : (re_y, re_x), 'leye' : (le_y, le_x)}``.
    """
    return self._database.annotations(file._f)
