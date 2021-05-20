#!/usr/bin/env python
# Yannick Dayer <yannick.dayer@idiap.ch>

from bob.bio.base.database import CSVDataset, CSVToSampleLoaderBiometrics
from bob.pipelines.sample_loaders import AnnotationsLoader
from bob.pipelines.sample import DelayedSample
from bob.db.base.annotations import read_annotation_file
from bob.extension.download import get_file
from bob.io.video import reader
from bob.extension import rc

from sklearn.pipeline import make_pipeline
import functools
import os.path
import logging
import numpy

logger = logging.getLogger(__name__)

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
        self.references_list = []

    def convert_row_to_sample(self, row, header):
        """Creates a sample given a row of the CSV protocol definition.
        """
        fields = dict([[str(h).lower(), r] for h, r in zip(header, row)])

        if self.reference_id_equal_subject_id:
            fields["subject_id"] = fields["reference_id"]
        else:
            if "subject_id" not in fields:
                raise ValueError(f"`subject_id` not available in {header}")
        if "should_flip" not in fields:
            raise ValueError(f"`should_flip` not available in {header}")
        if "purpose" not in fields:
            raise ValueError(f"`purpose` not available in {header}")

        kwargs = {k: fields[k] for k in fields.keys() - {"id","should_flip"}}

        # Retrieve the references list
        if fields["purpose"].lower() == "enroll" and fields["reference_id"] not in self.references_list:
            self.references_list.append(fields["reference_id"])
        # Set the references list in the probes for vanilla-biometrics
        if fields["purpose"].lower() != "enroll":
            if fields["attack_type"]:
                # Attacks are only compare to their target (no `spoof_neg`)
                kwargs["references"] = [fields["reference_id"]]
            else:
                kwargs["references"] = self.references_list
        # One row leads to multiple samples (different frames)
        return DelayedSample(
            functools.partial(
                load_frame_from_file_replaymobile,
                file_name=os.path.join(self.dataset_original_directory, fields["path"] + self.extension),
                frame=int(fields["frame"]),
                should_flip=fields["should_flip"]=="TRUE",
            ),
            key=fields['id'],
            **kwargs,
        )


def read_frame_annotation_file_replaymobile(file_name, frame, annotations_type="json"):
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

    video_annotations = read_annotation_file(file_name, annotation_type=annotations_type)
    # read_annotation_file returns an ordered dict with str keys as frame number
    # Annotations can be "null". Take the last annotated frame in this case
    offset = 1
    frame_annotations = video_annotations[f"{frame}"]
    if frame_annotations is None:
        logger.warning(f"Annotation for file '{file_name}' at frame {frame} was 'null' retrieving nearest frame's annotations.")
    while frame_annotations is None:
        frame_annotations = video_annotations[f"{max(frame-offset, 1)}"]
        if frame_annotations is not None:
            break
        frame_annotations = video_annotations[f"{min(frame+offset, len(video_annotations)-1)}"]
        offset += 1
        if frame-offset < 1 and frame+offset > len(video_annotations):
            raise IOError(f"Annotations file '{file_name}' does not contain any annotations.")
    return frame_annotations

class FrameBoundingBoxAnnotationLoader(AnnotationsLoader):
    """A transformer that adds bounding-box to a sample from annotations files.

    Parameters
    ----------

    annotation_directory: str or None
    """
    def __init__(self,
        annotation_directory=None,
        annotation_extension=".json",
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
            # Adds the annotations as delayed_attributes, loading them when needed
            annotated_samples.append(
                DelayedSample(
                    x._load,
                    parent=x,
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
        The protocol to use

    protocol_definition_path: str or None
        Specifies a path to download the database definition to.
        If None: Downloads and uses the ``bob_data_folder`` config.
        (See :py:func:`bob.extension.download.get_file`)

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
        protocol_name="grandtest",
        protocol_definition_path=None,
        data_path=None,
        data_extension=".mov",
        annotations_path=None,
        annotations_extension=".json",
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

        if annotations_path is None:
            name = "annotations-replaymobile-mtcnn-9cd6e452.tar.xz"
            annotations_path = get_file(
                name,
                [f"http://www.idiap.ch/software/bob/data/bob/bob.pad.face/{name}"],
                cache_subdir="annotations",
                file_hash="9cd6e452",
            )

        logger.info(f"Database: Loading database definition from '{protocol_definition_path}'.")
        logger.info(f"Database: Defining data files path as '{data_path}'.")
        logger.info(f"Database: Defining annotation files path as '{annotations_path}'.")
        super().__init__(
            protocol_definition_path,
            protocol_name,
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
            fetch_probes=False,
            **kwargs
        )
        self.annotation_type = "eyes-center"
        self.fixed_positions = None
