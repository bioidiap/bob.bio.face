#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
  PolaThermal database: database implementation
"""

from sklearn.pipeline import make_pipeline

import bob.io.base

from bob.bio.base.database import CSVDataset, CSVToSampleLoaderBiometrics
from bob.bio.face.database.sample_loaders import EyesAnnotations
from bob.extension import rc
from bob.extension.download import get_file


class PolaThermalDatabase(CSVDataset):
    """
    Collected by USA Army, the Polarimetric Thermal Database contains basically VIS and Thermal face images.

    Follow bellow the description of the imager used to capture this device.

    The **polarimetric** LWIR imager used to collect this database was developed by Polaris Sensor Technologies.
    The imager is based on the division-of-time spinning achromatic retarder (SAR) design that uses a spinning phase-retarder mounted in series with a linear wire-grid polarizer.
    This system, also referred to as a polarimeter, has a spectral response range of 7.5-11.1, using a Stirling-cooled mercury telluride focal plane array with pixel array dimensions of 640×480.
    A Fourier modulation technique is applied to the pixel readout, followed by a series expansion and inversion to compute the Stokes images.
    Data were recorded at 60 frames per second (fps) for this database, using a wide FOV of 10.6°×7.9°. Prior to collecting data for each subject, a two-point non-uniformity correction (NUC) was performed using a Mikron blackbody at 20°C and 40°C, which covers the range of typical facial temperatures (30°C-35°C).
    Data was recorded on a laptop using custom vendor software.

    An array of four Basler Scout series cameras was used to collect the corresponding **visible spectrum imagery**.
    Two of the cameras are monochrome (model # scA640-70gm), with pixel array dimensions of 659×494.
    The other two cameras are color (model # scA640-70gc), with pixel array dimensions of 658×494.


    The dataset contains 60 subjects in total.
    For **VIS** images (considered only the 87 pixels interpupil distance) there are 4 samples per subject with neutral expression (called baseline condition **B**) and 12 samples per subject varying the facial expression (called expression **E**).
    Such variability was introduced by asking the subject to count orally.
    In total there are 960 images for this modality.
    For the **thermal** images there are 4 types of thermal imagery based on the Stokes parameters (:math:`S_0`, :math:`S_1`, :math:`S_2` and :math:`S_3`) commonly used to represent the polarization state.
    The thermal imagery is the following:

    - :math:`S_0`: The conventional thermal image
    - :math:`S_1`
    - :math:`S_2`
    - DoLP: The degree-of-linear-polarization (DoLP) describes the portion of an electromagnetic wave that is linearly polarized, as defined :math:`\\frac{sqrt(S_{1}^{2} + S_{2}^{2})}{S_0}`.

    Since :math:`S_3` is very small and usually taken to be zero, the authors of the database decided not to provide this part of the data.
    The same facial expression variability introduced in **VIS** is introduced for **Thermal** images.
    The distance between the subject and the camera is  the last source of variability introduced in the thermal images.
    There are 3 ranges: R1 (2.5m), R2 (5m) and R3 (7.5m).
    In total there are 11,520 images for this modality and for each subject they are split as the following:

    +----------------+----------+----------+----------+
    | Imagery/Range  | R1 (B/E) | R2 (B/E) | R3 (B/E) |
    +================+==========+==========+==========+
    | :math:`S_0`    | 16 (8/8) | 16 (8/8) | 16 (8/8) |
    +----------------+----------+----------+----------+
    | :math:`S_1`    | 16 (8/8) | 16 (8/8) | 16 (8/8) |
    +----------------+----------+----------+----------+
    | :math:`S_2`    | 16 (8/8) | 16 (8/8) | 16 (8/8) |
    +----------------+----------+----------+----------+
    | DoLP           | 16 (8/8) | 16 (8/8) | 16 (8/8) |
    +----------------+----------+----------+----------+



    .. warning::
        Use the command below to set the path of the real data::

            $ bob config set bob.db.pola-thermal.directory [PATH-TO-MEDS-DATA]



    Parameters
    ----------

    protocol: str
        One of the database protocols.
    """

    def __init__(
        self, protocol, annotation_type="eyes-center", fixed_positions=None
    ):

        # Downloading model if not exists
        urls = PolaThermalDatabase.urls()
        filename = get_file(
            "polathermal.tar.gz",
            urls,
            file_hash="4693149bc883debe5a9e1441a4f5f4ae",
        )

        directory = rc.get("bob.db.pola-thermal.directory", "")

        def load(path):
            """
            Images in this dataset are stored as 16-bit PNG [0-65535]
            and bob.bio.face assumes images are between 0 and 255,
            so we divide by 257: 65535 / 255 = 257
            """
            return bob.io.base.load(path) / 257

        super().__init__(
            name="polathermal",
            protocol=protocol,
            dataset_protocol_path=filename,
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoaderBiometrics(
                    data_loader=load,
                    dataset_original_directory=directory,
                    extension=".png",
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
            "VIS-VIS-split1",
            "VIS-VIS-split2",
            "VIS-VIS-split3",
            "VIS-VIS-split4",
            "VIS-VIS-split5",
            "VIS-thermal-overall-split1",
            "VIS-thermal-overall-split2",
            "VIS-thermal-overall-split3",
            "VIS-thermal-overall-split4",
            "VIS-thermal-overall-split5",
            "VIS-polarimetric-overall-split1",
            "VIS-polarimetric-overall-split2",
            "VIS-polarimetric-overall-split3",
            "VIS-polarimetric-overall-split4",
            "VIS-polarimetric-overall-split5",
            "VIS-thermal-expression-split1",
            "VIS-thermal-expression-split2",
            "VIS-thermal-expression-split3",
            "VIS-thermal-expression-split4",
            "VIS-thermal-expression-split5",
            "VIS-polarimetric-expression-split1",
            "VIS-polarimetric-expression-split2",
            "VIS-polarimetric-expression-split3",
            "VIS-polarimetric-expression-split4",
            "VIS-polarimetric-expression-split5",
            "VIS-thermal-R1-split1",
            "VIS-thermal-R1-split2",
            "VIS-thermal-R1-split3",
            "VIS-thermal-R1-split4",
            "VIS-thermal-R1-split5",
            "VIS-polarimetric-R1-split1",
            "VIS-polarimetric-R1-split2",
            "VIS-polarimetric-R1-split3",
            "VIS-polarimetric-R1-split4",
            "VIS-polarimetric-R1-split5",
            "VIS-thermal-R2-split1",
            "VIS-thermal-R2-split2",
            "VIS-thermal-R2-split3",
            "VIS-thermal-R2-split4",
            "VIS-thermal-R2-split5",
            "VIS-polarimetric-R2-split1",
            "VIS-polarimetric-R2-split2",
            "VIS-polarimetric-R2-split3",
            "VIS-polarimetric-R2-split4",
            "VIS-polarimetric-R2-split5",
            "VIS-thermal-R3-split1",
            "VIS-thermal-R3-split2",
            "VIS-thermal-R3-split3",
            "VIS-thermal-R3-split4",
            "VIS-thermal-R3-split5",
            "VIS-polarimetric-R3-split1",
            "VIS-polarimetric-R3-split2",
            "VIS-polarimetric-R3-split3",
            "VIS-polarimetric-R3-split4",
            "VIS-polarimetric-R3-split5",
        ]

    @staticmethod
    def urls():
        return [
            "https://www.idiap.ch/software/bob/databases/latest/polathermal.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/polathermal.tar.gz",
        ]
