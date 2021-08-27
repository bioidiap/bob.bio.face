#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
  Multipie database implementation 
"""

from bob.bio.base.database import CSVDataset
from bob.bio.base.database import CSVToSampleLoaderBiometrics
from bob.bio.face.database.sample_loaders import EyesAnnotations
from bob.extension import rc
from bob.extension.download import get_file
import bob.io.base
from sklearn.pipeline import make_pipeline


class ARFaceDatabase(CSVDataset):
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

            bob config set bob.bio.face.arface.directory [ARFACE PATH]


    .. code-block:: latex

        @article{martinez1998ar,
        title={The AR Face Database: CVC Technical Report, 24},
        author={Martinez, Aleix and Benavente, Robert},
        year={1998}
        }


    """

    def __init__(self, protocol, annotation_type="eyes-center", fixed_positions=None):

        # Downloading model if not exists
        urls = ARFaceDatabase.urls()
        filename = get_file(
            "arface-d24d4413.tar.gz",
            urls,
            file_hash="71944cca12819003e3629da9c1cb8b66",
        )

        super().__init__(
            name="arface",
            dataset_protocol_path=filename,
            protocol=protocol,
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoaderBiometrics(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=rc["bob.bio.face.arface.directory"]
                    if rc["bob.bio.face.arface.directory"]
                    else "",
                    extension=".ppm",
                ),
                EyesAnnotations(),
            ),
            annotation_type=annotation_type,
            fixed_positions=fixed_positions,
        )

    @staticmethod
    def protocols():
        # TODO: Until we have (if we have) a function that dumps the protocols, let's use this one.
        protocols = [
            "all",
            "expression",
            "illumination",
            "occlusion",
            "occlusion_and_illumination",
        ]

    @staticmethod
    def urls():
        return [
            "https://www.idiap.ch/software/bob/databases/latest/arface-d24d4413.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/arface-d24d4413.tar.gz",
        ]
