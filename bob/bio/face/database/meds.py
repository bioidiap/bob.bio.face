#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
  MEDS database implementation 
"""

from bob.bio.base.database import (
    CSVDataset,
    CSVDatasetZTNorm,
)
from bob.bio.base.database import CSVToSampleLoaderBiometrics
from bob.bio.face.database.sample_loaders import EyesAnnotations
from bob.extension import rc
from bob.extension.download import get_file
import bob.io.base
from sklearn.pipeline import make_pipeline

# from bob.bio.face.database.sample_loaders import eyes_annotations_loader
import os


class MEDSDatabase(CSVDatasetZTNorm):
    """
    The MEDS II database was developed by NIST to support and assists their biometrics evaluation program.
    It is composed by 518 identities from both men/women (labeled as M and F) and five different race annotations (Asian, Black, American Indian, Unknown and White)
    (labeled as A, B, I, U and W.
    
    Unfortunately, the distribution of gender and race is extremely unbalanced as it can be
    observed in their statistics. Furthermore, only 256 subjects has
    more than one image sample (obviously it is not possible to do a biometric evaluation with one sample per subject).
    For this reason, this interface contains a subset of the data, which is composed only by 383 subjects (White and Black men only).

    This dataset contains three verification protocols and they are:
    `verification_fold1`, `verification_fold2` and `verification_fold1`.
    Follow below the identities distribution in each set for the for each protocol:


    +--------------------+---------------+-----------+-----------+-----------+
    |                    | Training set              | Dev. Set  | Eval. Set |
    +--------------------+---------------+-----------+           +           +
    |                    | T-References  | Z-Probes  |           |           |
    +====================+===============+===========+===========+===========+
    | verification_fold1 |      80       |     80    |   111     |   112     |
    +--------------------+---------------+-----------+-----------+-----------+
    | verification_fold2 |      80       |     80    |   111     |   112     |
    +--------------------+---------------+-----------+-----------+-----------+
    | verification_fold3 |      80       |     80    |   111     |   112     |
    +--------------------+---------------+-----------+-----------+-----------+
    
    Example
    -------

    Fetching biometric references::

    >>> from bob.bio.face.database import MEDSDatabase
    >>> database = MEDSDatabase(protocol="verification_fold1")
    >>> database.references()


    Fetching probes::

    >>> from bob.bio.face.database import MEDSDatabase
    >>> database = MEDSDatabase(protocol="verification_fold1")
    >>> database.probes()


    Fetching refererences for T-Norm normalization::

    >>> from bob.bio.face.database import MEDSDatabase
    >>> database = MEDSDatabase(protocol="verification_fold1")
    >>> database.trerefences()


    Fetching probes for Z-Norm normalization::

    >>> from bob.bio.face.database import MEDSDatabase
    >>> database = MEDSDatabase(protocol="verification_fold1")
    >>> database.zprobes()


    .. warning::
        Use the command below to set the path of the real data::

            $ bob config set bob.db.meds.directory [PATH-TO-MEDS-DATA]

    Parameters
    ----------

    protocol: str
        One of the database protocols. Options are `verification_fold1`, `verification_fold2` and `verification_fold3`

    """

    def __init__(self, protocol):

        # Downloading model if not exists
        urls = MEDSDatabase.urls()
        filename = get_file(
            "meds.tar.gz", urls, file_hash="3b01354d4c170672ac14120b80dace75"
        )

        self.annotation_type = "eyes-center"
        self.fixed_positions = None

        database = CSVDataset(
            filename,
            protocol,
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoaderBiometrics(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=rc["bob.db.meds.directory"]
                    if rc["bob.db.meds.directory"]
                    else "",
                    extension=".jpg",
                ),
                EyesAnnotations(),
            ),
        )

        super().__init__(database)

    @staticmethod
    def urls():
        return [
            "https://www.idiap.ch/software/bob/databases/latest/meds_v2.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/meds_v2.tar.gz",
        ]
