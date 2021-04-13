#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from .database import FaceBioFile
from .mobio import MobioDatabase
from .replay import ReplayBioDatabase
from .atnt import AtntBioDatabase
from .gbu import GBUBioDatabase
from .arface import ARFaceBioDatabase
from .lfw import LFWBioDatabase
from .multipie import MultipieDatabase
from .ijbc import IJBCBioDatabase
from .replaymobile import ReplayMobileBioDatabase
from .fargo import FargoBioDatabase
from .meds import MEDSDatabase
from .morph import MorphDatabase
from .casia_africa import CasiaAfricaDatabase
from .pola_thermal import PolaThermalDatabase
from .cbsr_nir_vis_2 import CBSRNirVis2Database

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
    MobioDatabase,
    ReplayBioDatabase,
    AtntBioDatabase,
    GBUBioDatabase,
    ARFaceBioDatabase,
    LFWBioDatabase,
    MultipieDatabase,
    ReplayMobileBioDatabase,
    FargoBioDatabase,
    MEDSDatabase,
    MorphDatabase,
    CasiaAfricaDatabase,
    PolaThermalDatabase,
    CBSRNirVis2Database,
)
__all__ = [_ for _ in dir() if not _.startswith("_")]
