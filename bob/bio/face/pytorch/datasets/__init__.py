#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# isort: skip_file

# gets sphinx autodoc done right - don't remove it

from .webface42m import WebFace42M
from .demographics import (
    MedsTorchDataset,
    MorphTorchDataset,
    MobioTorchDataset,
    MSCelebTorchDataset,
    VGG2TorchDataset,
    SiameseDemographicWrapper,
)


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
    WebFace42M,
    MedsTorchDataset,
    MorphTorchDataset,
    MobioTorchDataset,
    MSCelebTorchDataset,
    VGG2TorchDataset,
    SiameseDemographicWrapper,
)
__all__ = [_ for _ in dir() if not _.startswith("_")]
