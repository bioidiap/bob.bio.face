#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from .database import FaceBioFile
from .mobio import MobioBioDatabase
from .replay import ReplayBioDatabase
from .atnt import AtntBioDatabase
from .banca import BancaBioDatabase
from .gbu import GBUBioDatabase
from .arface import ARFaceBioDatabase
from .caspeal import CaspealBioDatabase
from .lfw import LFWBioDatabase
from .multipie import MultipieBioDatabase
from .ijba import IJBABioDatabase
from .xm2vts import XM2VTSBioDatabase
from .frgc import FRGCBioDatabase
from .scface import SCFaceBioDatabase
from .replaymobile import ReplayMobileBioDatabase
from .msu_mfsd_mod import MsuMfsdModBioDatabase

# gets sphinx autodoc done right - don't remove it


def __appropriate__(*args):
  """Says object was actually declared here, and not in the import module.
  Fixing sphinx warnings of not being able to find classes, when path is shortened.
  Parameters:

    *args: An iterable of objects to modify

  Resolves `Sphinx referencing issues
  <https://github.com/sphinx-doc/sphinx/issues/3048>`
  """

  for obj in args:
    obj.__module__ = __name__

__appropriate__(
  FaceBioFile,
  MobioBioDatabase,
  ReplayBioDatabase,
  AtntBioDatabase,
  BancaBioDatabase,
  GBUBioDatabase,
  ARFaceBioDatabase,
  CaspealBioDatabase,
  LFWBioDatabase,
  MultipieBioDatabase,
  IJBABioDatabase,
  XM2VTSBioDatabase,
  FRGCBioDatabase,
  SCFaceBioDatabase,
  ReplayMobileBioDatabase,
  MsuMfsdModBioDatabase,
)
__all__ = [_ for _ in dir() if not _.startswith('_')]
