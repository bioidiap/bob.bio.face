#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
  AR Face database implementation
"""

from clapper.rc import UserDefaults
from sklearn.pipeline import make_pipeline

import bob.io.base

from bob.bio.base.database import CSVDatabase, FileSampleLoader
from bob.bio.face.database.sample_loaders import EyesAnnotations

rc = UserDefaults("bobrc.toml")


class ARFaceDatabase(CSVDatabase):
    """
    This package contains the access API and descriptions for the AR face database.
    It only contains the Bob_ accessor methods to use the DB directly from python, with our certified protocols.
    The actual raw data for the database should be downloaded from the original URL (though we were not able to contact the corresponding Professor).

    Our version of the AR face database contains 3312 images from 136 persons, 76 men and 60 women.
    We split the database into several protocols that we have designed ourselves.
    The identities are split up into three groups:

    * the 'world' group for training your algorithm
    * the 'dev' group to optimize your algorithm parameters on
    * the 'eval' group that should only be used to report results

    Additionally, there are different protocols:

    * ``'expression'``: only the probe files with different facial expressions are selected
    * ``'illumination'``: only the probe files with different illuminations are selected
    * ``'occlusion'``: only the probe files with normal illumination and different accessories (scarf, sunglasses) are selected
    * ``'occlusion_and_illumination'``: only the probe files with strong illumination and different accessories (scarf, sunglasses) are selected
    * ``'all'``: all files are used as probe

    In any case, the images with neutral facial expression, neutral illumination and without accessories are used for enrollment.


    .. warning::

      To use this dataset protocol, you need to have the original files of the Mobio dataset.
      Once you have it downloaded, please run the following command to set the path for Bob

        .. code-block:: sh

            bob config set bob.bio.face.arface.directory [ARFACE DATA PATH]


    .. code-block:: latex

        @article{martinez1998ar,
        title={The AR Face Database: CVC Technical Report, 24},
        author={Martinez, Aleix and Benavente, Robert},
        year={1998}
        }


    """

    name = "arface"
    category = "face"
    dataset_protocols_name = "arface.tar.gz"
    dataset_protocols_urls = [
        "https://www.idiap.ch/software/bob/databases/latest/face/arface-7078bd96.tar.gz",
        "http://www.idiap.ch/software/bob/databases/latest/face/arface-7078bd96.tar.gz",
    ]
    dataset_protocols_hash = "7078bd96"

    def __init__(
        self, protocol, annotation_type="eyes-center", fixed_positions=None
    ):
        super().__init__(
            name=self.name,
            protocol=protocol,
            transformer=make_pipeline(
                FileSampleLoader(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=rc.get(
                        "bob.bio.face.arface.directory", ""
                    ),
                    extension=rc.get("bob.bio.face.arface.extension", ".ppm"),
                ),
                EyesAnnotations(),
            ),
            annotation_type=annotation_type,
            fixed_positions=fixed_positions,
        )
