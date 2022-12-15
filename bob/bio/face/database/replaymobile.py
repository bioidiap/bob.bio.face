#!/usr/bin/env python
# Yannick Dayer <yannick.dayer@idiap.ch>

import functools
import logging
import os.path

import imageio
import numpy

from sklearn.pipeline import make_pipeline

import bob.io.image

from bob.bio.base.database import CSVDataset, CSVToSampleLoaderBiometrics
from bob.bio.base.utils.annotations import read_annotation_file
from bob.extension import rc
from bob.extension.download import get_file
from bob.pipelines import hash_string
from bob.pipelines.sample import DelayedSample
from bob.pipelines.sample_loaders import AnnotationsLoader

logger = logging.getLogger(__name__)

read_annotation_file = functools.lru_cache()(read_annotation_file)


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
    logger.debug(f"Reading frame {frame} from '{file_name}'")

    video_reader = imageio.get_reader(file_name)
    image = video_reader.get_data(frame)
    # Convert to bob format (channel, height, width)
    image = bob.io.image.to_bob(image)

    # Image captured by the 'mobile' device are flipped vertically.
    # (Images were captured horizontally and bob.io.video does not read the
    #   metadata correctly, whether it was on the right or left side)
    if not should_flip:
        # after changing from bob.io.video to imageio-ffmpeg, turns out tablet
        # videos should be flipped to match previous behavior.
        image = numpy.flip(image, 2)

    return image


class ReplayMobileCSVFrameSampleLoader(CSVToSampleLoaderBiometrics):
    """A loader transformer returning a specific frame of a video file.

    This is specifically tailored for replay-mobile. It uses a specific loader
    that processes the `should_flip` metadata to correctly orient the frames.
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
        """Creates a sample given a row of the CSV protocol definition."""
        fields = dict([[str(h).lower(), r] for h, r in zip(header, row)])

        if self.reference_id_equal_subject_id:
            fields["subject_id"] = fields["reference_id"]
        else:
            if "subject_id" not in fields:
                raise ValueError(f"`subject_id` not available in {header}")
        if "should_flip" not in fields:
            raise ValueError(f"`should_flip` not available in {header}")

        kwargs = {k: fields[k] for k in fields.keys() - {"id", "should_flip"}}

        # One row creates one samples (=> one comparison because of `is_sparse`)
        should_flip = fields["should_flip"].lower() == "true"
        return DelayedSample(
            functools.partial(
                load_frame_from_file_replaymobile,
                file_name=os.path.join(
                    self.dataset_original_directory,
                    fields["path"] + self.extension,
                ),
                frame=int(fields["frame"]),
                should_flip=should_flip,
            ),
            key=fields["id"],
            should_flip=should_flip,
            **kwargs,
        )


def read_frame_annotation_file_replaymobile(
    file_name, frame, annotations_type="json"
):
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
        The annotation file name (relative to annotations_path).

    frame: int
        The video frame index.
    """
    logger.debug(f"Reading annotation file '{file_name}', frame {frame}.")

    video_annotations = read_annotation_file(
        file_name, annotation_type=annotations_type
    )
    # read_annotation_file returns an ordered dict with str keys as frame number
    frame_annotations = video_annotations[str(frame)]
    if frame_annotations is None:
        logger.warning(
            f"Annotation for file '{file_name}' at frame {frame} was 'null'."
        )
    return frame_annotations


class FrameBoundingBoxAnnotationLoader(AnnotationsLoader):
    """A transformer that adds bounding-box to a sample from annotations files.

    Parameters
    ----------

    annotation_directory: str or None
    """

    def __init__(
        self, annotation_directory=None, annotation_extension=".json", **kwargs
    ):
        super().__init__(
            annotation_directory=annotation_directory,
            annotation_extension=annotation_extension,
            **kwargs,
        )

    def transform(self, X):
        """Adds the bounding-box annotations to a series of samples."""
        if self.annotation_directory is None:
            return None

        annotated_samples = []
        for x in X:
            # Adds the annotations as delayed_attributes, loading them when needed
            annotated_samples.append(
                DelayedSample.from_sample(
                    x,
                    delayed_attributes=dict(
                        annotations=functools.partial(
                            read_frame_annotation_file_replaymobile,
                            file_name=f"{self.annotation_directory}:{x.path}{self.annotation_extension}",
                            frame=int(x.frame),
                            annotations_type=self.annotation_type,
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
        The protocol to use. Must be a sub-folder of ``protocol_definition_path``

    protocol_definition_path: str or None
        Specifies a path where to fetch the database definition from.
        (See :py:func:`bob.extension.download.get_file`)
        If None: Downloads the file in the path from ``bob_data_folder`` config.
        If None and the config does not exist: Downloads the file in ``~/bob_data``.

    data_path: str or None
        Overrides the config-defined data location.
        If None: uses the ``bob.db.replaymobile.directory`` config.
        If None and the config does not exist, set as cwd.

    annotation_path: str or None
        Specifies a path where the annotation files are located.
        If None: Downloads the files to the path poited by the
        ``bob.db.replaymobile.annotation_directory`` config.
        If None and the config does not exist: Downloads the file in ``~/bob_data``.
    """

    def __init__(
        self,
        protocol="grandtest",
        protocol_definition_path=None,
        data_path=None,
        data_extension=".mov",
        annotations_path=None,
        annotations_extension=".json",
        **kwargs,
    ):

        if protocol_definition_path is None:
            # Downloading database description files if it is not specified
            proto_def_hash = "fee57d46"
            proto_def_name = f"replaymobile-{proto_def_hash}.tar.gz"
            proto_def_urls = [
                f"https://www.idiap.ch/software/bob/databases/latest/{proto_def_name}",
                f"http://www.idiap.ch/software/bob/databases/latest//{proto_def_name}",
            ]
            protocol_definition_path = get_file(
                filename=proto_def_name,
                urls=proto_def_urls,
                cache_subdir="datasets",
                file_hash=proto_def_hash,
            )

        if data_path is None:
            data_path = rc.get("bob.db.replaymobile.directory", "")
        if data_path == "":
            logger.warning(
                "Raw data path is not configured. Please set "
                "'bob.db.replaymobile.directory' with the 'bob config set' command. "
                "Will now attempt with current directory."
            )

        if annotations_path is None:
            annot_hash = "9cd6e452"
            annot_name = f"annotations-replaymobile-mtcnn-{annot_hash}.tar.xz"
            annot_urls = [
                f"https://www.idiap.ch/software/bob/data/bob/bob.pad.face/{annot_name}",
                f"http://www.idiap.ch/software/bob/data/bob/bob.pad.face/{annot_name}",
            ]
            annotations_path = get_file(
                filename=annot_name,
                urls=annot_urls,
                cache_subdir="annotations",
                file_hash=annot_hash,
                extract=True,
            )
            annotations_path = os.path.join(
                os.path.dirname(annotations_path),
                "replaymobile-mtcnn-annotations",
            )

        logger.info(
            f"Database: Will read CSV protocol definitions in '{protocol_definition_path}'."
        )
        logger.info(f"Database: Will read raw data files in '{data_path}'.")
        logger.info(
            f"Database: Will read annotation files in '{annotations_path}'."
        )
        super().__init__(
            name="replaymobile",
            protocol=protocol,
            dataset_protocol_path=protocol_definition_path,
            csv_to_sample_loader=make_pipeline(
                ReplayMobileCSVFrameSampleLoader(
                    dataset_original_directory=data_path,
                    extension=data_extension,
                ),
                FrameBoundingBoxAnnotationLoader(
                    annotation_directory=annotations_path,
                    annotation_extension=annotations_extension,
                ),
            ),
            is_sparse=True,
            **kwargs,
        )
        self.annotation_type = "eyes-center"
        self.fixed_positions = None
        self.hash_fn = hash_string
