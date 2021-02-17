#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
  MOBIO database implementation 
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


class MobioDatabase(CSVDatasetZTNorm):
    """
    The MOBIO dataset is a video database containing bimodal data (face/speaker).
    It is composed by 152 people (split in the two genders male and female), mostly Europeans, split in 5 sessions (few weeks time lapse between sessions).
    The database was recorded using two types of mobile devices: mobile phones (NOKIA N93i) and laptop 
    computers(standard 2008 MacBook).

    For face recognition images are used instead of videos.
    One image was extracted from each video by choosing the video frame after 10 seconds.
    The eye positions were manually labelled and distributed with the database.

    For more information check:

    .. code-block:: latex

        @article{McCool_IET_BMT_2013,
            title = {Session variability modelling for face authentication},
            author = {McCool, Chris and Wallace, Roy and McLaren, Mitchell and El Shafey, Laurent and Marcel, S{\'{e}}bastien},
            month = sep,
            journal = {IET Biometrics},
            volume = {2},
            number = {3},
            year = {2013},
            pages = {117-129},
            issn = {2047-4938},
            doi = {10.1049/iet-bmt.2012.0059},
        }

    """

    def __init__(self, protocol):

        # Downloading model if not exists
        urls = MobioDatabase.urls()
        filename = get_file(
            "mobio.tar.gz", urls, file_hash="42cee778c17a34762d5fc5dd13ce3ee6"
        )

        self.annotation_type = "eyes-center"
        self.fixed_positions = None

        database = CSVDataset(
            filename,
            protocol,
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoaderBiometrics(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=rc["bob.db.mobio.directory"]
                    if rc["bob.db.mobio.directory"]
                    else "",
                    extension=".png",
                ),
                EyesAnnotations(),
            ),
        )

        super().__init__(database)

    @staticmethod
    def protocols():
        # TODO: Until we have (if we have) a function that dumps the protocols, let's use this one.
        return [
            "laptop1-female",
            "laptop_mobile1-female",
            "mobile0-female",
            "mobile0-male-female",
            "mobile1-male",
            "laptop1-male",
            "laptop_mobile1-male",
            "mobile0-male",
            "mobile1-female",
        ]

    @staticmethod
    def urls():
        return [
            "https://www.idiap.ch/software/bob/databases/latest/mobio_v2.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/mobio_v2.tar.gz",
        ]
