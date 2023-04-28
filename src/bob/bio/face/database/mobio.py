#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
  MOBIO database implementation
"""

from clapper.rc import UserDefaults
from sklearn.pipeline import make_pipeline

import bob.io.base

from bob.bio.base.database import CSVDatabase, FileSampleLoader
from bob.bio.face.database.sample_loaders import EyesAnnotations

rc = UserDefaults("bobrc.toml")


class MobioDatabase(CSVDatabase):
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

    name = "mobio"
    category = "face"
    dataset_protocols_name = "mobio.tar.gz"
    dataset_protocols_urls = [
        "https://www.idiap.ch/software/bob/databases/latest/face/mobio-0580d95a.tar.gz",
        "http://www.idiap.ch/software/bob/databases/latest/face/mobio-0580d95a.tar.gz",
    ]
    dataset_protocols_hash = "0580d95a"

    def __init__(
        self,
        protocol,
        annotation_type="eyes-center",
        fixed_positions=None,
        dataset_original_directory=rc.get("bob.db.mobio.directory", ""),
        dataset_original_extension=rc.get("bob.db.mobio.extension", ".png"),
    ):
        super().__init__(
            name=self.name,
            protocol=protocol,
            transformer=make_pipeline(
                FileSampleLoader(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=dataset_original_directory,
                    extension=dataset_original_extension,
                ),
                EyesAnnotations(),
            ),
            templates_metadata=["gender"],
            annotation_type=annotation_type,
            fixed_positions=fixed_positions,
        )
