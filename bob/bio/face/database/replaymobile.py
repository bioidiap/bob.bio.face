#!/usr/bin/env python
# Yannick Dayer <yannick.dayer@idiap.ch>

from bob.pipelines.datasets import FileListDatabase, CSVToSamples
from bob.pipelines.sample import DelayedSample
from bob.extension.download import list_dir, search_file
from bob.db.base.utils import check_parameters_for_validity
from bob.db.base.annotations import read_annotation_file
from bob.io.video import reader as video_reader
from bob.bio.base.pipelines.vanilla_biometrics import Database

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import make_pipeline
import functools
import os.path
import logging
import numpy

logger = logging.getLogger(__name__)

class VideoReader(TransformerMixin, BaseEstimator):
    """Transformer that loads the video data from a file
    """
    def __init__(self, data_path, file_ext=".mov"):
        self.data_path = data_path
        self.file_ext = file_ext

    def transform(self, X, y=None):
        all_samples = []
        for sample in X:
            all_samples.append(
                DelayedSample(
                    load=functools.partial(video_reader, os.path.join(self.data_path, sample.PATH + self.file_ext)),
                    parent=sample,
                )
            )
        return all_samples

    def fit(self, X, y=None):
        return self

    def _more_tags(self):
        return {
            "stateless": True,
            "requires_fit": False,
        }


def get_frame_from_sample(video_sample, frame_id):
    """Returns one frame's data from a replay-mobile video sample.

    Flips the image according to the sample's metadata.
    """
    frame = video_sample.data[frame_id]
    if video_sample.SHOULD_FLIP: # TODO include this field in the csv files
        frame = numpy.flip(frame, 2)
    # Convert to bob format (channel, height, width)
    frame = numpy.transpose(frame, (0, 2, 1))
    return frame

class VideoToFrames(TransformerMixin, BaseEstimator):
    """Transformer that creates a list of frame samples from a video sample.

    Parameters
    ----------

    frame_indices: None or Sequence[int]
        The list of frames to keep. Will keep all the frames if None or empty.
    """
    def __init__(self, frame_indices=None):
        super().__init__()
        self.frame_indices = frame_indices

    def transform(self, X, y=None):
        all_samples = []
        # Iterate over each video sample
        for video_sample in X:
            # Extract frames from the file
            [
                all_samples.append(DelayedSample(
                    load=functools.partial(get_frame_from_sample, video_sample, frame_id),
                    parent=video_sample,
                    frame=frame_id,
                    key=f"{video_sample.ID}_{frame_id}")
                )
                for frame_id in range(len(video_sample.data))
                if not self.frame_indices or frame_id in self.frame_indices
            ]
        return all_samples

    def fit(self, X, y=None):
        return self

    def _more_tags(self):
        return {
            "stateless": True,
            "requires_fit": False,
        }


def read_frame_annotations_file(file_name, frame_id):
    """Reads an annotations file and extracts one frame's annotations.
    """
    video_annotations = read_annotation_file(file_name, annotation_type="json")
    # read_annotation_file returns an ordered dict with string keys
    return video_annotations[f"{frame_id}"]

class AnnotationsAdder(TransformerMixin, BaseEstimator):
    """Transformer that adds an 'annotations' field to the samples.

    This reads a json file containing coordinates for each frame of a video.
    """
    def __init__(self, annotation_directory):
        self.annotation_directory=annotation_directory

    def transform(self, X, y=None):
        all_samples = []
        for sample in X:
            delayed_annotations = functools.partial(
                read_frame_annotations_file,
                file_name=f"{self.annotation_directory}:{sample.PATH}.json",
                frame_id=sample.frame,
            )
            all_samples.append(
                DelayedSample(
                    load=sample._load,
                    parent=sample,
                    delayed_attributes = {"annotations": delayed_annotations},
                )
            )
        return all_samples

    def fit(self, X, y=None):
        return self

    def _more_tags(self):
        return {
            "stateless": True,
            "requires_fit": False,
        }


class CSVToBioSamples(CSVToSamples):
    """Iterator that reads a CSV and creates Samples.
    """
    def __iter__(self):
        for sample in super().__iter__():
            # TODO test that fields are present? (attack_type for vuln?)
            yield sample


class ReplayMobileBioDatabase(FileListDatabase, Database):
    """Database for Replay-mobile-img for vulnerability analysis
    """
    def __init__(
        self,
        dataset_protocols_path,
        protocol,
        data_path,
        data_extension=".mov",
        annotations_path=None,
        **kwargs,
    ):
        super().__init__(
            dataset_protocols_path,
            protocol,
            reader_cls=CSVToBioSamples,
            transformer=make_pipeline(
                VideoReader(data_path=data_path, file_ext=data_extension),
                VideoToFrames(range(12,251,24)),
                AnnotationsAdder(annotations_path),
            ),
            **kwargs
        )
        self.annotations_path = self.dataset_protocols_path if not annotations_path else annotations_path # TODO default to protocol_path?
        self.annotation_type = "eyes-center"
        self.fixed_positions = None

    def groups(self):
        names = list_dir(self.dataset_protocols_path, self.protocol, files=False)
        names = [os.path.splitext(n)[0] for n in names]
        return names

    def list_file(self, group, purpose):
        if purpose == "enroll":
            purpose_name = "for_models"
        elif purpose == "probe":
            purpose_name = "for_probes"
        elif purpose == "train":
            purpose_name = "train_world"
        else:
            raise ValueError(f"Unknown purpose '{purpose}'.")
        # Protocol files are in the form <db_name>/{dev,eval,train}/{for_models,for_probes}.csv
        list_file = search_file(
            self.dataset_protocols_path,
            os.path.join(self.protocol, group, purpose_name + ".csv"),
        )
        return list_file

    def get_reader(self, group, purpose): # TODO use the standard csv format instead?
        key = (self.protocol, group, purpose)
        if key not in self.readers:
            self.readers[key] = self.reader_cls(
                list_file=self.list_file(group, purpose), transformer=self.transformer
            )

        reader = self.readers[key]
        return reader

    def samples(self, groups, purpose):
        groups = check_parameters_for_validity(
            groups, "groups", self.groups(), self.groups()
        )
        all_samples = []
        for grp in groups:

            for sample in self.get_reader(grp, purpose):
                all_samples.append(sample)

        return all_samples

    def background_model_samples(self):
        return self.samples(groups="train", purpose="train")

    def references(self, group):
        return self.samples(groups=group, purpose="enroll")

    def probes(self, group):
        return self.samples(groups=group, purpose="probe")

    def all_samples(self, groups):
        return super().all_samples(groups=groups)
