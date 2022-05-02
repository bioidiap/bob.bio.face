#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
  MOBIO database implementation
"""

from sklearn.pipeline import make_pipeline

import bob.io.base

from bob.bio.base.database import CSVDatasetZTNorm, CSVToSampleLoaderBiometrics
from bob.bio.face.database.sample_loaders import EyesAnnotations
from bob.extension import rc
from bob.extension.download import get_file


class MobioDatabase(CSVDatasetZTNorm):
    """
    The MOBIO dataset is a video database containing bimodal data (face/speaker).
    It is composed by 152 people (split in the two genders male and female), mostly Europeans, split in 5 sessions (few weeks time lapse between sessions).
    The database was recorded using two types of mobile devices: mobile phones (NOKIA N93i) and laptop
    computers(standard 2008 MacBook).

    For face recognition images are used instead of videos.
    One image was extracted from each video by choosing the video frame after 10 seconds.
    The eye positions were manually labelled and distributed with the database.


    .. warning::

      To use this dataset protocol, you need to have the original files of the Mobio dataset.
      Once you have it downloaded, please run the following command to set the path for Bob

        .. code-block:: sh

            bob config set bob.db.mobio.directory [MOBIO PATH]

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

    def __init__(
        self,
        protocol,
        annotation_type="eyes-center",
        fixed_positions=None,
        dataset_original_directory=rc.get("bob.db.mobio.directory", ""),
        dataset_original_extension=rc.get("bob.db.mobio.extension", ".png"),
    ):

        # Downloading model if not exists
        urls = MobioDatabase.urls()
        filename = get_file(
            "mobio.tar.gz", urls, file_hash="4a7f99b33a54b2dd337ddcaecb09edb8"
        )

        super().__init__(
            name="mobio",
            dataset_protocol_path=filename,
            protocol=protocol,
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoaderBiometrics(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=dataset_original_directory,
                    extension=dataset_original_extension,
                ),
                EyesAnnotations(),
            ),
            annotation_type=annotation_type,
            fixed_positions=fixed_positions,
        )

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
            "https://www.idiap.ch/software/bob/databases/latest/mobio-7fdf4f20.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/mobio-7fdf4f20.tar.gz",
        ]
