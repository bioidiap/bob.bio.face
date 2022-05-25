#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Thu May 24 10:41:42 CEST 2012
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import os

import pytest

import bob.bio.base
import bob.extension.log

from bob.extension import rc
from bob.extension.download import get_file

logger = bob.extension.log.setup("bob.bio.face")


def _check_annotations(
    database, topleft=False, required=True, limit_files=None, framed=False
):
    database_legacy = database.database
    files = database_legacy.all_files()
    if limit_files is not None:
        import random

        files = random.sample(files, limit_files)
    found_none = False

    for file in files:
        annotations = database_legacy.annotations(file)
        if required:
            assert annotations is not None
        if annotations is not None:
            assert isinstance(annotations, dict)
            if framed:
                # take one of the frames
                annotations = annotations[list(annotations.keys())[0]]
            if topleft:
                assert "topleft" in annotations
                assert "bottomright" in annotations
            else:
                assert "reye" in annotations
                assert "leye" in annotations
        else:
            found_none = True
    if found_none:
        logger.warn(
            "Some annotations were None for {}".format(database_legacy.name)
        )


def test_mobio():
    from bob.bio.face.database import MobioDatabase

    # Getting the absolute path
    urls = MobioDatabase.urls()
    filename = get_file("mobio.tar.gz", urls)

    # Removing the file before the test
    try:
        os.remove(filename)
    except Exception:
        pass

    protocols = MobioDatabase.protocols()
    for p in protocols:
        database = MobioDatabase(protocol=p)
        assert len(database.background_model_samples()) > 0
        assert len(database.treferences()) > 0
        assert len(database.zprobes()) > 0

        assert len(database.references(group="dev")) > 0
        assert len(database.probes(group="dev")) > 0

        assert len(database.references(group="eval")) > 0
        assert len(database.probes(group="eval")) > 0

    # Sanity check on mobio-male
    database = MobioDatabase(protocol="mobile0-male")
    assert len(database.treferences()) == 24
    assert len(database.zprobes()) == 960
    assert len(database.background_model_samples()) == 9600

    assert len(database.references()) == 24
    assert len(database.probes()) == 2520

    assert len(database.references(group="eval")) == 38
    assert len(database.probes(group="eval")) == 3990


def test_multipie():
    from bob.bio.face.database import MultipieDatabase

    # Getting the absolute path
    urls = MultipieDatabase.urls()
    filename = get_file("multipie.tar.gz", urls)

    # Removing the file before the test
    try:
        os.remove(filename)
    except Exception:
        pass

    protocols = MultipieDatabase.protocols()

    for p in protocols:
        database = MultipieDatabase(protocol=p)
        assert len(database.background_model_samples()) > 0

        assert len(database.references(group="dev")) > 0
        assert len(database.probes(group="dev")) > 0

        assert len(database.references(group="eval")) > 0
        assert len(database.probes(group="eval")) > 0

    database = MultipieDatabase(protocol="P")
    assert len(database.background_model_samples()) == 7725

    assert len(database.references(group="dev")) == 64
    assert len(database.probes(group="dev")) == 3328

    assert len(database.references(group="eval")) == 65
    assert len(database.probes(group="eval")) == 3380


def test_replaymobile():
    database = bob.bio.base.load_resource(
        "replaymobile-img", "database", preferred_package="bob.bio.face"
    )
    sample = database.probes()[0][0]
    assert (
        sample.path
        == "devel/real/client005_session02_authenticate_mobile_adverse"
    ), sample.path
    assert sample.frame == "12", sample.frame
    assert sample.should_flip, sample
    assert hasattr(sample, "annotations")
    assert "reye" in sample.annotations
    assert "leye" in sample.annotations
    assert hasattr(sample, "path")
    assert hasattr(sample, "frame")
    assert len(database.references()) == 16
    assert len(database.references(group="eval")) == 12
    assert len(database.probes()) == 4160
    assert len(database.probes(group="eval")) == 3020
    assert sample.annotations == {
        "bottomright": [734, 407],
        "topleft": [436, 182],
        "leye": [541, 350],
        "reye": [540, 245],
        "mouthleft": [655, 254],
        "mouthright": [657, 338],
        "nose": [591, 299],
    }

    # test another sample where should_flip is False
    sample2 = [s for s in database.all_samples() if not s.should_flip][0]
    assert (
        sample2.path
        == "enroll/train/client001_session01_enroll_tablet_lightoff"
    ), sample2.path
    assert sample2.frame == "12", sample2.frame
    assert not sample2.should_flip, sample2
    assert sample2.annotations == {
        "reye": [515, 267],
        "leye": [516, 399],
        "nose": [576, 332],
        "mouthleft": [662, 282],
        "mouthright": [664, 384],
        "topleft": [372, 196],
        "bottomright": [761, 480],
    }, dict(sample2.annotations)

    # Only if data is available
    if rc.get("bob.db.replaymobile.directory", None):
        assert sample.data.shape == (3, 1280, 720), sample.data.shape
        assert sample.data[0, 0, 0] == 94, sample.data[0, 0, 0]

        assert sample2.data.shape == (3, 1280, 720), sample2.data.shape
        assert sample2.data[0, 0, 0] == 129, sample2.data[0, 0, 0]


@pytest.mark.skipif(
    rc.get("bob.bio.face.ijbc.directory") is None,
    reason="IJBC original protocols not available. Please do `bob config set bob.bio.face.ijbc.directory [IJBC PATH]` to set the IJBC data path.",
)
@pytest.mark.slow
def test_ijbc():
    from bob.bio.face.database import IJBCDatabase

    # test1 #####

    database = IJBCDatabase(protocol="test1")

    # assert len(database.background_model_samples()) == 140732
    assert len(database.references()) == 3531
    assert len(database.probes()) == 19593
    num_comparisons = sum([len(item.references) for item in database.probes()])
    assert num_comparisons == 19557 + 15638932  # Genuine + Impostor

    # test2 #####
    database = IJBCDatabase(protocol="test2")

    assert len(database.references()) == 140739
    assert len(database.probes()) == 140739

    num_comparisons = sum([len(item.references) for item in database.probes()])
    assert num_comparisons == 39208203

    # test4 G1 #####
    database = IJBCDatabase(protocol="test4-G1")

    assert len(database.references()) == 1772
    assert len(database.probes()) == 19593
    num_comparisons = sum([len(item.references) for item in database.probes()])
    assert num_comparisons == 34718796

    # test4 G2 #####
    database = IJBCDatabase(protocol="test4-G2")

    assert len(database.references()) == 1759
    assert len(database.probes()) == 19593
    num_comparisons = sum([len(item.references) for item in database.probes()])
    assert num_comparisons == 34464087


def test_meds():

    from bob.bio.face.database import MEDSDatabase

    # Getting the absolute path
    urls = MEDSDatabase.urls()
    filename = get_file("meds.tar.gz", urls)

    # Removing the file before the test
    try:
        os.remove(filename)
    except Exception:
        pass

    database = MEDSDatabase("verification_fold1")

    assert len(database.background_model_samples()) == 234
    assert len(database.references()) == 111
    assert len(database.probes()) == 313

    assert len(database.zprobes()) == 80
    assert len(database.treferences()) == 80

    assert len(database.references(group="eval")) == 112
    assert len(database.probes(group="eval")) == 309


def test_morph():

    from bob.bio.face.database import MorphDatabase

    # Getting the absolute path
    urls = MorphDatabase.urls()
    filename = get_file("morph.tar.gz", urls)

    # Removing the file before the test
    try:
        os.remove(filename)
    except Exception:
        pass

    database = MorphDatabase("verification_fold1")

    assert len(database.background_model_samples()) == 226
    assert len(database.references()) == 6738
    assert len(database.probes()) == 6557

    assert len(database.zprobes()) == 66
    assert len(database.treferences()) == 69

    assert len(database.references(group="eval")) == 6742
    assert len(database.probes(group="eval")) == 6553


def test_casia_africa():

    from bob.bio.face.database import CasiaAfricaDatabase

    database = CasiaAfricaDatabase("ID-V-All-Ep1")

    assert len(database.references()) == 2455
    assert len(database.probes()) == 2426


def test_frgc():

    from bob.bio.face.database import FRGCDatabase

    def _check_samples(samples, n_templates, n_subjects, template_size=0):
        assert len(samples) == n_templates
        assert len(set([x.reference_id for x in samples])) == n_templates
        assert len(set([x.subject_id for x in samples])) == n_subjects

        if template_size > 0:
            assert all([len(x) == template_size for x in samples])

    database = FRGCDatabase("2.0.1")
    _check_samples(
        database.background_model_samples(), n_templates=12776, n_subjects=222
    )
    _check_samples(
        database.references(), n_templates=7572, n_subjects=370, template_size=1
    )
    _check_samples(
        database.probes(), n_templates=8456, n_subjects=345, template_size=1
    )

    database = FRGCDatabase("2.0.2")
    _check_samples(
        database.background_model_samples(), n_templates=12776, n_subjects=222
    )
    _check_samples(
        database.references(), n_templates=1893, n_subjects=370, template_size=4
    )
    _check_samples(
        database.probes(), n_templates=2114, n_subjects=345, template_size=4
    )

    database = FRGCDatabase("2.0.4")
    _check_samples(
        database.background_model_samples(), n_templates=12776, n_subjects=222
    )
    _check_samples(
        database.references(), n_templates=7572, n_subjects=370, template_size=1
    )
    _check_samples(
        database.probes(), n_templates=4228, n_subjects=345, template_size=1
    )


def test_polathermal():

    from bob.bio.face.database import PolaThermalDatabase

    database = PolaThermalDatabase("VIS-thermal-overall-split1")
    assert len(database.references()) == 35
    assert len(database.probes()) == 1680


@pytest.mark.skipif(
    rc.get("bob.bio.face.rfw.directory") is None,
    reason="RFW original protocols not available. Please do `bob config set bob.bio.face.rfw.directory [RFW PATH]` to set the RFW data path.",
)
def test_rfw():

    from bob.bio.face.database import RFWDatabase

    database = RFWDatabase("original")
    assert len(database.references()) == 22481
    assert len(database.probes()) == 21851

    database = RFWDatabase("idiap")
    assert len(database.references()) == 22481
    assert len(database.probes()) == 21851
    assert len(database.zprobes()) == 100
    assert len(database.treferences()) == 100
    assert sum([len(d.references) for d in database.probes()]) == 89540


def test_scface():
    from bob.bio.face.database import SCFaceDatabase

    # Getting the absolute path
    urls = SCFaceDatabase.urls()
    filename = get_file("scface.tar.gz", urls)

    # Removing the file before the test
    try:
        os.remove(filename)
    except Exception:
        pass

    N_WORLD, N_DEV, N_EVAL = 43, 44, 43

    N_WORLD_DISTANCES = 3
    N_RGB_CAMS, N_IR_CAMS = 5, 2
    N_RGB_MUGSHOTS, N_IR_MUGSHOTS = 1, 1

    def _check_protocol(p, n_mugshots, n_cams, n_distances):
        database = SCFaceDatabase(protocol=p)

        assert len(database.background_model_samples()) == N_WORLD * (
            n_mugshots + N_WORLD_DISTANCES * n_cams
        )

        assert len(database.references(group="dev")) == N_DEV * n_mugshots
        assert len(database.probes(group="dev")) == N_DEV * n_distances * n_cams

        assert len(database.references(group="eval")) == N_EVAL * n_mugshots
        assert (
            len(database.probes(group="eval")) == N_EVAL * n_distances * n_cams
        )

        return p

    checked_protocols = []
    checked_protocols.append(
        _check_protocol("combined", N_RGB_MUGSHOTS, N_RGB_CAMS, n_distances=3)
    )
    checked_protocols.append(
        _check_protocol("close", N_RGB_MUGSHOTS, N_RGB_CAMS, n_distances=1)
    )
    checked_protocols.append(
        _check_protocol("medium", N_RGB_MUGSHOTS, N_RGB_CAMS, n_distances=1)
    )
    checked_protocols.append(
        _check_protocol("far", N_RGB_MUGSHOTS, N_RGB_CAMS, n_distances=1)
    )
    checked_protocols.append(
        _check_protocol("IR", N_IR_MUGSHOTS, N_IR_CAMS, n_distances=1)
    )

    for p in SCFaceDatabase.protocols():
        assert p in checked_protocols, "Protocol {} untested".format(p)


def test_cbsr_nir_vis_2():

    from bob.bio.face.database import CBSRNirVis2Database

    database = CBSRNirVis2Database("view2_1")

    assert len(database.references()) == 358
    assert len(database.probes()) == 6123


@pytest.mark.skipif(
    rc.get("bob.bio.face.gbu.directory") is None,
    reason="GBU original protocols not available. Please do `bob config set bob.bio.face.gbu.directory [GBU PATH]` to set the GBU data path.",
)
def test_gbu():

    from bob.bio.face.database import GBUDatabase

    database = GBUDatabase("Good")
    assert len(database.references()) == 1085
    assert len(database.probes()) == 1085

    database = GBUDatabase("Bad")
    assert len(database.references()) == 1085
    assert len(database.probes()) == 1085

    database = GBUDatabase("Ugly")
    assert len(database.references()) == 1085
    assert len(database.probes()) == 1085

    assert len(database.background_model_samples()) == 3910


def test_caspeal():

    from bob.bio.face.database import CaspealDatabase

    # protocols = ['accessory', 'aging', 'background', 'distance', 'expression', 'lighting', 'pose']

    database = CaspealDatabase("accessory")
    assert len(database.references()) == 1040
    assert len(database.probes()) == 2285
    assert len(database.background_model_samples()) == 1200

    database = CaspealDatabase("aging")
    assert len(database.references()) == 1040
    assert len(database.probes()) == 66
    assert len(database.background_model_samples()) == 1200

    database = CaspealDatabase("background")
    assert len(database.references()) == 1040
    assert len(database.probes()) == 553
    assert len(database.background_model_samples()) == 1200

    database = CaspealDatabase("distance")
    assert len(database.references()) == 1040
    assert len(database.probes()) == 275
    assert len(database.background_model_samples()) == 1200

    database = CaspealDatabase("expression")
    assert len(database.references()) == 1040
    assert len(database.probes()) == 1570
    assert len(database.background_model_samples()) == 1200

    database = CaspealDatabase("lighting")
    assert len(database.references()) == 1040
    assert len(database.probes()) == 2243
    assert len(database.background_model_samples()) == 1200

    # There's no pose protocol


def test_arface():
    def assert_attributes(samplesets, attribute, references):
        assert sorted(
            set([getattr(sset, attribute) for sset in samplesets])
        ) == sorted(references)

    from bob.bio.face.database import ARFaceDatabase

    # protocols = 'all','expression', 'illumination', 'occlusion', 'occlusion_and_illumination'
    # expression_choices = ("neutral", "smile", "anger", "scream")
    # illumination_choices = ("front", "left", "right", "all")
    # occlusion_choices = ("none", "sunglasses", "scarf")
    # session_choices = ("first", "second")

    database = ARFaceDatabase("all")
    assert len(database.references(group="dev")) == 43
    assert len(database.probes(group="dev")) == 1032
    assert len(database.references(group="eval")) == 43
    assert len(database.probes(group="eval")) == 1032
    assert len(database.background_model_samples()) == 1076

    database = ARFaceDatabase("expression")
    assert len(database.references(group="dev")) == 43
    assert len(database.probes(group="dev")) == 258
    assert len(database.references(group="eval")) == 43
    assert len(database.probes(group="eval")) == 258
    for group in ["dev", "eval"]:
        assert_attributes(
            database.probes(group=group),
            attribute="expression",
            references=("smile", "anger", "scream"),
        )
        assert_attributes(
            database.probes(group="dev"),
            attribute="illumination",
            references=("front",),
        )
        assert_attributes(
            database.probes(group="dev"),
            attribute="occlusion",
            references=("none",),
        )

    assert len(database.background_model_samples()) == 1076

    database = ARFaceDatabase("illumination")
    assert len(database.references(group="dev")) == 43
    assert len(database.probes(group="dev")) == 258
    assert len(database.references(group="eval")) == 43
    assert len(database.probes(group="eval")) == 258
    assert len(database.background_model_samples()) == 1076
    for group in ["dev", "eval"]:
        assert_attributes(
            database.probes(group=group),
            attribute="expression",
            references=("neutral",),
        )
        assert_attributes(
            database.probes(group="dev"),
            attribute="illumination",
            references=("left", "right", "all"),
        )
        assert_attributes(
            database.probes(group="dev"),
            attribute="occlusion",
            references=("none",),
        )

    database = ARFaceDatabase("occlusion")
    assert len(database.references(group="dev")) == 43
    assert len(database.probes(group="dev")) == 172
    assert len(database.references(group="eval")) == 43
    assert len(database.probes(group="eval")) == 172
    assert len(database.background_model_samples()) == 1076
    for group in ["dev", "eval"]:
        assert_attributes(
            database.probes(group=group),
            attribute="expression",
            references=("neutral",),
        )
        assert_attributes(
            database.probes(group="dev"),
            attribute="illumination",
            references=("front",),
        )
        assert_attributes(
            database.probes(group="dev"),
            attribute="occlusion",
            references=("sunglasses", "scarf"),
        )

    database = ARFaceDatabase("occlusion_and_illumination")
    assert len(database.references(group="dev")) == 43
    assert len(database.probes(group="dev")) == 344
    assert len(database.references(group="eval")) == 43
    assert len(database.probes(group="eval")) == 344
    assert len(database.background_model_samples()) == 1076

    for group in ["dev", "eval"]:
        assert_attributes(
            database.probes(group=group),
            attribute="expression",
            references=("neutral",),
        )
        assert_attributes(
            database.probes(group="dev"),
            attribute="illumination",
            references=("left", "right"),
        )
        assert_attributes(
            database.probes(group="dev"),
            attribute="occlusion",
            references=("sunglasses", "scarf"),
        )


@pytest.mark.skipif(
    rc.get("bob.bio.face.lfw.directory") is None,
    reason="LFW original protocols not available. Please do `bob config set bob.bio.face.lfw.directory [LFW PATH]` to set the LFW data path.",
)
def test_lfw():

    from bob.bio.face.database import LFWDatabase

    # protocol view2
    database = LFWDatabase(protocol="view2")
    references = database.references()
    probes = database.probes()
    assert len(references) == 4564
    assert len(probes) == 4576
    assert len(database.background_model_samples()) == 0
    # We need to have 6000 comparisons
    assert sum([len(p.references) for p in probes]) == 6000

    # Funneled annotations
    database = LFWDatabase(protocol="view2", annotation_issuer="funneled")
    references = database.references()
    assert references[0][0].annotations["leye"] == (114.086957, 131.5434785)

    # Idiap annotations
    database = LFWDatabase(protocol="view2", annotation_issuer="idiap")
    references = database.references()
    assert references[0][0].annotations["leye"] == (139.0, 107.0)

    database = LFWDatabase(protocol="view2", annotation_issuer="named")
    references = database.references()
    assert references[0][0].annotations["leye"] == (114.0, 132.0)

    # protocol o1
    database = LFWDatabase(protocol="o1")
    references = database.references()
    probes = database.probes()

    assert len(references) == 610
    assert len(probes) == 6264

    # check training set
    training = database.background_model_samples()
    assert len(training) == 611
    assert len(training[-1].samples) == 1070

    # check that we enroll from 3 images each and probe with 1
    assert all(len(r.samples) == 3 for r in references)
    assert all(len(p.samples) == 1 for p in probes)

    # check the number of known and unknown probe samples
    assert sum(1 for p in probes if p.subject_id != "unknown") == 4903, sum(
        1 for p in probes if p.subject_id != "unknown"
    )
    assert sum(1 for p in probes if p.subject_id == "unknown") == 1361, sum(
        1 for p in probes if p.subject_id == "unknown"
    )

    # protocol o2
    database = LFWDatabase(protocol="o2")
    references = database.references()
    probes = database.probes()

    # check the number of known and unknown probe samples
    assert sum(1 for p in probes if p.subject_id != "unknown") == 4903, sum(
        1 for p in probes if p.subject_id != "unknown"
    )
    assert sum(1 for p in probes if p.subject_id == "unknown") == 4069, sum(
        1 for p in probes if p.subject_id == "unknown"
    )

    # protocol o3
    database = LFWDatabase(protocol="o3")
    references = database.references()
    probes = database.probes()

    # check the number of known and unknown probe samples
    assert sum(1 for p in probes if p.subject_id != "unknown") == 4903, sum(
        1 for p in probes if p.subject_id != "unknown"
    )
    assert sum(1 for p in probes if p.subject_id == "unknown") == 5430, sum(
        1 for p in probes if p.subject_id == "unknown"
    )


def test_vgg2():
    from bob.bio.face.database import VGG2Database

    # Getting the absolute path
    urls = VGG2Database.urls()
    filename = get_file("vgg2.tar.gz", urls)

    # Removing the file before the test
    try:
        os.remove(filename)
    except Exception:
        pass

    p = "vgg2-short"

    database = VGG2Database(protocol=p)

    # Sanity check on vgg2-short
    assert len(database.treferences()) == 194
    assert len(database.zprobes()) == 200

    # vgg2-full has 3'141'890 SAMPLES
    assert len(database.background_model_samples()) == 86310
    assert len(database.references()) == 500
    assert len(database.probes()) == 2500

    p = "vgg2-short-with-eval"

    database = VGG2Database(protocol=p)

    # Sanity check on vgg2-short
    assert len(database.treferences()) == 194
    assert len(database.zprobes()) == 200

    # vgg2-full has 3'141'890 SAMPLES
    assert len(database.background_model_samples()) == 86310
    assert len(database.references(group="dev")) == 250
    assert len(database.probes(group="dev")) == 1250

    assert len(database.references(group="eval")) == 250
    assert len(database.probes(group="eval")) == 1250
