#!/usr/bin/env python
# Yannick Dayer <yannick.dayer@idiap.ch>

from bob.bio.base.database import CSVDataset, CSVToSampleLoaderBiometrics
from bob.pipelines.datasets.sample_loaders import AnnotationsLoader
from bob.pipelines.sample import DelayedSample
from bob.extension.download import get_file
from bob.io.video import reader
from bob.extension import rc
import bob.core

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import make_pipeline
import functools
import os.path
import numpy

logger = bob.core.log.setup("bob.bio.face")

def load_frame_from_file_replaymobile(file_name, frame, should_flip):
    """Loads a single frame from a video file for replay-mobile.

    This function uses bob's video reader utility that does not load the full
    video in memory to just access one frame.

    Parameters
    ----------

    file_name: str
        The video file to load the frames from

    frame: None or list of int
        The index of the frame to load.

    capturing device: str
        'mobile' devices' frames will be flipped vertically.
        Other devices' frames will not be flipped.

    Returns
    -------

    images: 3D numpy array
        The frame of the video in bob format (channel, height, width)
    """
    logger.debug(f"Extracting frame {frame} from '{file_name}'")
    video_reader = reader(file_name)
    image = video_reader[frame]
    # Image captured by the 'mobile' device are flipped vertically.
    # (Images were captured horizontally and bob.io.video does not read the
    #   metadata correctly, whether it was on the right or left side)
    if should_flip:
        image = numpy.flip(image, 2)
    # Convert to bob format (channel, height, width)
    image = numpy.transpose(image, (0, 2, 1))
    return image

def read_frame_annotation_file_replaymobile(file_name, frame):
    """Returns the bounding-box for one frame of a video file of replay-mobile.

    Given an annnotation file location and a frame number, returns the bounding
    box coordinates corresponding to the frame.

    The replay-mobile annotation files are composed of 4 columns and N rows for
    N frames of the video:

    120 230 40 40
    125 230 40 40
    ...
    <x> <y> <w> <h>

    Parameters
    ----------

    file_name: str
        The complete annotation file path and name (with extension).

    frame: int
        The video frame index.
    """
    logger.debug(f"Reading annotation file '{file_name}', frame {frame}.")
    if not file_name:
        return None

    if not os.path.exists(file_name):
        raise IOError(f"The annotation file '{file_name}' was not found")

    with open(file_name, 'r') as f:
        # One line is one frame, each line contains a bounding box coordinates
        line = f.readlines()[frame]

    positions = line.split(' ')

    if len(positions) != 4:
        raise ValueError(f"The content of '{file_name}' was not correct for frame {frame} ({positions})")

    annotations = {
        'topleft': (float(positions[1]), float(positions[0])),
        'bottomright':(
            float(positions[1])+float(positions[3]),
            float(positions[0])+float(positions[2])
        )
    }

    return annotations

class ReplayMobileCSVFrameSampleLoader(CSVToSampleLoaderBiometrics):
    """A loader transformer returning a specific frame of a video file.

    This is specifically tailored for replay-mobile. It uses a specific loader
    that takes the capturing device as input.
    """
    def __init__(
        self,
        dataset_original_directory="",
        extension="",
        reference_id_equal_subject_id=True,
    ):
        super().__init__(
            data_loader=None,
            extension=extension,
            dataset_original_directory=dataset_original_directory,
        )
        self.reference_id_equal_subject_id = reference_id_equal_subject_id

    def convert_row_to_sample(self, row, header):
        """Creates a set of samples given a row of the CSV protocol definition.
        """
        path = row[0]
        reference_id = row[1]
        id = row[2] # Will be used as 'key'

        kwargs = dict([[str(h).lower(), r] for h, r in zip(header[3:], row[3:])])
        if self.reference_id_equal_subject_id:
            kwargs["subject_id"] = reference_id
        else:
            if "subject_id" not in kwargs:
                raise ValueError(f"`subject_id` not available in {header}")
        if "should_flip" not in kwargs:
            raise ValueError(f"`should_flip` not available in {header}")
        # One row leads to multiple samples (different frames)
        all_samples = [DelayedSample(
            functools.partial(
                load_frame_from_file_replaymobile,
                file_name=os.path.join(self.dataset_original_directory, path + self.extension),
                frame=frame,
                should_flip=kwargs["should_flip"]=="TRUE",
            ),
            key=f"{id}_{frame}",
            path=path,
            reference_id=reference_id,
            frame=frame,
            **kwargs,
        ) for frame in range(12,251,24)]
        return all_samples


class FrameBoundingBoxAnnotationLoader(AnnotationsLoader):
    """A transformer that adds bounding-box to a sample from annotations files.

    Parameters
    ----------

    annotation_directory: str or None
    """
    def __init__(self,
        annotation_directory=None,
        annotation_extension=".face",
        **kwargs
    ):
        super().__init__(
            annotation_directory=annotation_directory,
            annotation_extension=annotation_extension,
            **kwargs
        )

    def transform(self, X):
        """Adds the bounding-box annotations to a series of samples.
        """
        if self.annotation_directory is None:
            return None

        annotated_samples = []
        for x in X:

            # Build the path to the annotation files structure
            annotation_file = os.path.join(
                self.annotation_directory, x.path + self.annotation_extension
            )

            annotated_samples.append(
                DelayedSample(
                    x._load,
                    parent=x,
                    delayed_attributes=dict(
                        annotations=functools.partial(
                            read_frame_annotation_file_replaymobile,
                            file_name=annotation_file,
                            frame=int(x.frame),
                        )
                    ),
                )
            )

        return annotated_samples

class ReplayMobileBioDatabase(CSVDataset):
    """Database interface that loads a csv definition for replay-mobile

    Looks for the protocol definition files (structure of CSV files). If not
    present, downloads them.
    Then sets the data and annotation paths from __init__ parameters or from
    the configuration (``bob config`` command).

    Parameters
    ----------

    protocol_name: str
        The protocol to use

    protocol_definition_path: str or None
        Specifies a path to download the database definition to.
        If None: Downloads and uses the ``bob_data_folder`` config.
        (See :py:fct:`bob.extension.download.get_file`)

    data_path: str or None
        Overrides the config-defined data location.
        If None: uses the ``bob.db.replaymobile.directory`` config.
        If None and the config does not exist, set as cwd.

    annotation_path: str or None
        Overrides the config-defined annotation files location.
        If None: uses the ``bob.db.replaymobile.annotation_directory`` config.
        If None and the config does not exist, set as
        ``{data_path}/faceloc/rect``.
    """
    def __init__(
        self,
        protocol_name="bio-grandtest",
        protocol_definition_path=None,
        data_path=None,
        annotation_path=None,
        **kwargs
    ):
        if protocol_definition_path is None:
            # Downloading database description files if it is not specified
            urls = [
                "https://www.idiap.ch/software/bob/databases/latest/replay-mobile-csv.tar.gz",
                "http://www.idiap.ch/software/bob/databases/latest/replay-mobile-csv.tar.gz",
            ]
            protocol_definition_path = get_file("replay-mobile-csv.tar.gz", urls)

        if data_path is None:
            # Defaults to cwd if config not defined
            data_path = rc.get("bob.db.replaymobile.directory", "")

        if annotation_path is None:
            # Defaults to {data_path}/faceloc/rect if config not defined
            annotation_path = rc.get(
                "bob.db.replaymobile.annotation_directory",
                os.path.join(data_path, "faceloc/rect/")
            )

        logger.info(f"Database: Loading database definition from '{protocol_definition_path}'.")
        logger.info(f"Database: Defining data files path as '{data_path}'.")
        logger.info(f"Database: Defining annotation files path as '{annotation_path}'.")
        super().__init__(
            protocol_definition_path,
            protocol_name,
            csv_to_sample_loader=make_pipeline(
                ReplayMobileCSVFrameSampleLoader(
                    dataset_original_directory=data_path,
                    extension=".mov",
                ),
                FrameBoundingBoxAnnotationLoader(
                    annotation_directory=annotation_path,
                    annotation_extension=".face",
                ),
            ),
            **kwargs
        )
        self.annotation_type = "bounding-box"
        self.fixed_positions = None
