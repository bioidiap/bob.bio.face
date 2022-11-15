from bob.bio.base.pipelines.abstract_classes import Database
from bob.extension import (rc, download)

import os
import glob
import pandas as pd
import scipy.io
from bob.pipelines import DelayedSample, SampleSet,sample_loaders
import bob.io.base
from functools import partial
from bob.io.image import (bob_to_opencvbgr, opencvbgr_to_bob)
import cv2

def preprocess_insightface(inputs, shape=(112,112), interpolation=cv2.INTER_AREA):
    bob_img = bob.io.base.load(inputs)
    cv2_img = bob_to_opencvbgr(bob_img)

    #1: manage grayscale
    if cv2_img.shape[-1] != 3:
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2BGR)

    #2: resize to 112x112, control interpolation
    cv2_resized = cv2.resize(cv2_img, shape, interpolation=interpolation)

    bob_resized = opencvbgr_to_bob(cv2_resized)

    return bob_resized

class QMULSurvFaceDatabase(Database):

    def __init__(self,
                 protocol='default'
                 ):
        self.original_directory = os.path.join(rc["bob.bio.face.survface.directory"])
        super().__init__(
            name='QMUL-SurvFace',
            protocol=protocol,
            score_all_vs_all=False)

        #read eval set
        probes_df, references_df, matches_df = self.__load_eval_set(os.path.join(self.original_directory, 'Face_Verification_Test_Set'))
        self.eval_references_df = references_df
        self.eval_probes_df = probes_df
        self.eval_matching = matches_df

        #read dev set
        dev_probes_df, dev_references_df = self.__load_dev_set()
        self.dev_probes_df = dev_probes_df
        self.dev_references_df = dev_references_df

    def background_model_samples(self):
        return []

    def references(self, group='eval'):
        assert group in ('dev', 'eval'), 'Only dev and eval sets are implemented'
        references_df = self.eval_references_df if group == 'eval' else self.dev_references_df
        return self.__make_samplesets(references_df, self.original_directory, group)

    def probes(self, group='eval'):
        assert group in ('dev', 'eval'), 'Only dev and eval sets are implemented'
        if group=='eval':
            # Reorganize list of matching to regroup them by probe template id
            probes_df = self.eval_probes_df
            indexed_matching = self.eval_matching.groupby('probe_reference_id')
        else:
            probes_df = self.dev_probes_df

        probes = self.__make_samplesets(probes_df, self.original_directory, group)

        for sampleset in probes:
            # Get the list of references to which this probe template should be compared
            if group == 'eval':
                compared_references = indexed_matching.get_group(sampleset.reference_id)
                # Add the list of reference template id under the `references` field of the sampleset
                sampleset.references = compared_references['bio_ref_reference_id'].tolist()

            else: #dev set
                sampleset.references = self.dev_references_df["reference_id"].tolist()  # implementing all vs all
        return probes

    def all_samples(self, group='eval'):
        assert group in ('dev', 'eval'), 'Only dev and eval sets are implemented'

        ## NB : this will contain duplicates as some samples appear both as probes and references
        return self.probes(group=group) + self.references(group=group)

    def groups(self):
        return ['eval', 'dev']

    def protocols(self):
        return ['default']

    def __load_eval_set(self, ROOT):
        # Load from .mat
        raw_pos_matching = scipy.io.matlab.loadmat(os.path.join(ROOT, 'positive_pairs_names.mat'))[
            'positive_pairs_names']
        raw_neg_matching = scipy.io.matlab.loadmat(os.path.join(ROOT, 'negative_pairs_names.mat'))[
            'negative_pairs_names']

        # Format loaded data, store as Dataframe
        def format_pair(pair):
            return str(pair[0][0]), str(pair[1][0])

        pos_matching = pd.DataFrame(map(format_pair, raw_pos_matching),
                                    columns=['probe_reference_id', 'bio_ref_reference_id'])
        neg_matching = pd.DataFrame(map(format_pair, raw_neg_matching),
                                    columns=['probe_reference_id', 'bio_ref_reference_id'])
        matching = pd.concat([pos_matching, neg_matching])

        files = list(set(matching['probe_reference_id'].tolist() + matching['bio_ref_reference_id'].tolist()))
        df = pd.DataFrame(
            {'key': [os.path.join('Face_Verification_Test_Set', 'verification_images', key) for key in files]})

        # Merge the Dataframe containing the filenames, with the generated dataframe containing all the metadata

        def decode_filename(filename_full):
            filename = filename_full.split('/')[-1]
            metadata, ext = os.path.splitext(filename)
            try:
                subject_id, camera_id, img_name = metadata.split('_')
            except Exception:
                # print(metadata)
                # Some files are missing the camera
                subject_id, img_name = metadata.split('_')
                camera_id = None

            return {'subject_id': subject_id, 'reference_id': filename, 'camera_id': camera_id}

        df = df.merge(pd.DataFrame.from_records(df['key'].apply(decode_filename)), left_index=True, right_index=True)

        is_probe = df['reference_id'].isin(matching['probe_reference_id'])
        is_reference = df['reference_id'].isin(matching['bio_ref_reference_id'])

        probes_df = df[is_probe]
        references_df = df[is_reference]

        return probes_df, references_df, matching

    def __make_samplesets(self, df, image_dir, group='eval'):
        # Utilitary function to turn a Dataframe into a list of SampleSets (= a list of biometric templates)

        # We group the samples by reference to form SampleSets
        templates = df.groupby('reference_id')

        # Utilitary function to turn 1 dataframe row into a DelayedSample

        def make_sample_from_row(row, image_dir):
            return DelayedSample(load=partial(preprocess_insightface, inputs=os.path.join(image_dir, row['key'])),
                                 **dict(row))
        subject_id_key = 'subject_id' if group == 'eval' else 'reference_id'
        samplesets = []
        for reference_id, samples in templates:
            subject_id = samples.iloc[0][subject_id_key]
            key = samples.iloc[0]['key']
            samplesets.append(SampleSet(samples.apply(make_sample_from_row, axis=1, image_dir=image_dir).tolist(),
                                        reference_id=reference_id if group == 'eval' else subject_id,
                                        subject_id=subject_id, key=key))

        return samplesets

    def __load_dev_set(self):
        urls = QMULSurvFaceDatabase.urls()
        filename = download.get_file(
            "QMUL-SurvFace.tar.gz",
            urls,
            file_hash="0f171acfc1eea1732cd73b6e042109c0",
        )

        for_models_csv = download.search_file(filename,
                                              [os.path.join(self.name, "default", "dev", "for_models.csv")])
        for_probes_csv = download.search_file(filename,
                                              [os.path.join(self.name, "default", "dev", "for_probes.csv")])
        references_df = pd.read_csv(for_models_csv)
        probes_df = pd.read_csv(for_probes_csv)
        references_df.rename(columns={'PATH': 'key', 'REFERENCE_ID': 'reference_id'}, inplace=True, errors="raise")
        probes_df.rename(columns={'PATH': 'key', 'REFERENCE_ID': 'reference_id'}, inplace=True, errors="raise")

        return probes_df, references_df

    @staticmethod
    def urls():
        return [
            "https://gitlab.idiap.ch/bob/bob.bio.face/-/tree/low_resolution/bob/bio/face/data",
            "https://gitlab.idiap.ch/bob/bob.bio.face/-/tree/low_resolution/bob/bio/face/data",
        ]
