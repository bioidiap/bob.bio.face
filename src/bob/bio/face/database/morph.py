#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
  MORPH database implementation
"""

from clapper.rc import UserDefaults
from sklearn.pipeline import make_pipeline

import bob.io.base

from bob.bio.base.database import CSVDatabase, FileSampleLoader
from bob.bio.face.database.sample_loaders import EyesAnnotations

rc = UserDefaults("bobrc.toml")


class MorphDatabase(CSVDatabase):
    """
    The MORPH dataset is relatively old, but is getting some traction recently mostly because its richness
    with respect to sensitive attributes.
    It is composed by 55,000 samples from 13,000 subjects from men and women and five
    race clusters (called ancestry) and they are the following: African, European, Asian, Hispanic and Others. Figure 8
    present some samples from this database.

    This dataset contains faces from five ethnicities (African, European, Asian, Hispanic, "Other")
    and two genders (Male and Female).
    Furthermore, this interface contains three verification protocols and they are:
    `verification_fold1`, `verification_fold2` and `verification_fold1`.
    Follow below the identities distribution in each set for the for each protocol:

    +--------------------+---------------+-----------+-----------+-----------+
    |                    | Training set              | Dev. Set  | Eval. Set |
    +--------------------+---------------+-----------+           +           +
    |                    | T-References  | Z-Probes  |           |           |
    +====================+===============+===========+===========+===========+
    | verification_fold1 |      69       |     66    |   6738    |   6742    |
    +--------------------+---------------+-----------+-----------+-----------+
    | verification_fold2 |      69       |     67    |   6734    |   6737    |
    +--------------------+---------------+-----------+-----------+-----------+
    | verification_fold3 |      70       |     66    |   6736    |   6740    |
    +--------------------+---------------+-----------+-----------+-----------+

    .. warning::
        Use the command below to set the path of the real data::

            $ bob config set bob.db.morph.directory [PATH-TO-MORPH-DATA]

    Parameters
    ----------

    protocol: str
        One of the database protocols. Options are `verification_fold1`, `verification_fold2` and `verification_fold3`

    """

    name = "morph"
    category = "face"
    dataset_protocols_name = "morph.tar.gz"
    dataset_protocols_urls = [
        "https://www.idiap.ch/software/bob/databases/latest/face/morph-1200b906.tar.gz",
        "http://www.idiap.ch/software/bob/databases/latest/face/morph-1200b906.tar.gz",
    ]
    dataset_protocols_hash = "1200b906"

    def __init__(
        self,
        protocol,
        annotation_type="eyes-center",
        fixed_positions=None,
        dataset_original_directory=rc.get("bob.db.morph.directory", ""),
        dataset_original_extension=rc.get("bob.db.morph.extension", ".JPG"),
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
            templates_metadata=["date_of_birth", "sex", "rac"],
            annotation_type=annotation_type,
            fixed_positions=fixed_positions,
        )
