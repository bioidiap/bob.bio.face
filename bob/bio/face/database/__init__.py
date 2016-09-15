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
from .cuhk_cufs import CUHK_CUFSBioDatabase
from .scface import SCFaceBioDatabase

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
