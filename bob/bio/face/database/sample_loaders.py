#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""  Sample loader"""


from bob.bio.base.database import CSVToSampleLoader
from bob.pipelines import Sample, DelayedSample, SampleSet
import functools
import os


class CSVToSampleLoaderEyesAnnotations(CSVToSampleLoader):
    """
    Convert CSV files in the format below to either a list of
    :any:`bob.pipelines.DelayedSample` or :any:`bob.pipelines.SampleSet`

    Convert  leye_x, leye_y, reye_x, reye_y attributes to `annotations = (leye, reye)`

    """

    def convert_row_to_sample(self, row, header):
        path = row[0]
        subject = row[1]
        kwargs = dict([[h, r] for h, r in zip(header[2:], row[2:])])

        annotations = {
            "leye": (kwargs["leye_x"], kwargs["leye_y"]),
            "reye": (kwargs["reye_x"], kwargs["reye_y"]),
        }

        kwargs.pop("leye_x")
        kwargs.pop("leye_y")
        kwargs.pop("reye_x")
        kwargs.pop("reye_y")

        return DelayedSample(
            functools.partial(
                self.data_loader,
                os.path.join(self.dataset_original_directory, path + self.extension),
            ),
            key=path,
            subject=subject,
            annotations=annotations,
            **kwargs,
        )


"""
class CSVToSampleLoaderEyesAnnotations(CSVToSampleLoader):
    def __call__(self, filename):
        import ipdb

        ipdb.set_trace()
        samples = super(CSVToSampleLoaderEyesAnnotations, self).__call__(filename)

        def generate_annotations(sample):
            
            Convert  leye_x, leye_y, reye_x, reye_y attributes to
            `annotations = (leye, reye)`

            

            check_keys = [
                a in (sample.__dict__.keys())
                for a in ["leye_x", "leye_y", "reye_x", "reye_y"]
            ]

            if not check_keys:
                raise ValueError(
                    "Sample needs to contain the following annotations: 'leye_x', 'leye_y', 'reye_x', 'reye_y'"
                )

            annotations = {
                "leye": (sample.leye_x, sample.leye_y),
                "reye": (sample.reye_x, sample.reye_y),
            }

            # Changing the state of samples for efficiency
            # We might have a gigantic amount of datasets
            sample.__dict__.pop("leye_x")
            sample.__dict__.pop("leye_y")
            sample.__dict__.pop("reye_x")
            sample.__dict__.pop("reye_y")
            sample.annotations = annotations

        for sample in samples:
            generate_annotations(sample)

        return samples
"""
