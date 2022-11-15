from bob.extension import rc
from bob.extension.download import get_file
from sklearn.pipeline import make_pipeline
import bob.io.base
from bob.bio.face.annotator import TinyFace as tinyface_annotator
import numpy as np
from skimage.transform import resize
from bob.io.image import (bob_to_opencvbgr, opencvbgr_to_bob)
import cv2

from bob.bio.base.database import (CSVDataset, CSVToSampleLoaderBiometrics)

def preprocess_insightface(path_str, shape=(112,112), interpolation=cv2.INTER_AREA):
    bob_img = bob.io.base.load(path_str)
    cv2_img = bob_to_opencvbgr(bob_img)

    #1: manage grayscale
    if cv2_img.shape[-1] != 3:
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2BGR)

    #2: resize to 112x112, control interpolation
    cv2_resized = cv2.resize(cv2_img, shape, interpolation=interpolation)

    #repack to bob
    bob_resized = opencvbgr_to_bob(cv2_resized)

    return bob_resized

class TinyFaceDatabase(CSVDataset):
    def __init__(self, protocol, **kwargs,):
        # Downloading model if not exists
        urls = TinyFaceDatabase.urls()
        filename = get_file(
            "tinyface.tar.gz",
            urls,
            file_hash="23586cd2b342a8dca6a89637847efd61", #This version of the file is 8000 samples for the test set
        )

        super().__init__(
            name="tinyface",
            dataset_protocol_path=filename,
            protocol=protocol,
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoaderBiometrics(
                    #  data_loader=bob.io.base.load,
                    data_loader=preprocess_insightface,
                    dataset_original_directory=rc[
                        "bob.bio.face.tinyface.directory"
                    ]
                    if rc["bob.bio.face.tinyface.directory"]
                    else "",
                    extension=rc["bob.bio.face.tinyface.extension"]
                    if rc["bob.bio.face.tinyface.extension"]
                    else "",
                ),
            ),
            annotation_type=None,
            fixed_positions=None,
            score_all_vs_all=True,
        )

    @staticmethod
    def protocols():
        # TODO: Until we have (if we have) a function that dumps the protocols, let's use this one.
        return [
            "id_split"
        ]

    @staticmethod
    def urls():
        return [
            "https://gitlab.idiap.ch/bob/bob.bio.face/-/raw/low_resolution/bob/bio/face/data/tinyface.tar.gz",
            "https://gitlab.idiap.ch/bob/bob.bio.face/-/raw/low_resolution/bob/bio/face/data/tinyface.tar.gz",
        ]

    def objects(
        self, model_ids=None, groups=None, purposes=None, protocol=None
    ):
        samples = []

        if groups is None:
            groups = self.groups()

        if "train" in groups or "world" in groups:
            samples += self.background_model_samples()

        if purposes is None:
            purposes = ("enroll", "probe")

        if "enroll" in purposes:
            samples += self.references()

        if "probe" in purposes:
            samples += self.probes()

        if model_ids:
            samples = [s for s in samples if s.reference_id in model_ids]

        # create the old attributes
        for s in samples:
            s.client_id = s.reference_id
            s.path = s.id = s.key

        return samples
