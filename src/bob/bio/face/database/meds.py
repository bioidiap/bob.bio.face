#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
  MEDS database implementation
"""

from clapper.rc import UserDefaults
from sklearn.pipeline import make_pipeline

import bob.io.base

from bob.bio.base.database import CSVDatabase, FileSampleLoader
from bob.bio.face.database.sample_loaders import EyesAnnotations

rc = UserDefaults("bobrc.toml")


class MEDSDatabase(CSVDatabase):
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


    Fetching references for T-Norm normalization::

    >>> from bob.bio.face.database import MEDSDatabase
    >>> database = MEDSDatabase(protocol="verification_fold1")
    >>> database.treferences()


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

    name = "meds"
    category = "face"
    dataset_protocols_name = "meds.tar.gz"
    dataset_protocols_urls = [
        "https://www.idiap.ch/software/bob/databases/latest/face/meds-b74e3c3a.tar.gz",
        "http://www.idiap.ch/software/bob/databases/latest/face/meds-b74e3c3a.tar.gz",
    ]
    dataset_protocols_hash = "b74e3c3a"

    def __init__(
        self,
        protocol,
        annotation_type="eyes-center",
        fixed_positions=None,
        dataset_original_directory=rc.get("bob.db.meds.directory", ""),
        dataset_original_extension=rc.get("bob.db.meds.extension", ".jpg"),
    ):
        super().__init__(
            name=self.name,
            protocol=protocol,
            transformer=make_pipeline(
                FileSampleLoader(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=dataset_original_directory
                    if dataset_original_directory
                    else "",
                    extension=dataset_original_extension,
                ),
                EyesAnnotations(),
            ),
            templates_metadata=["sex", "race"],
            annotation_type=annotation_type,
            fixed_positions=fixed_positions,
        )
