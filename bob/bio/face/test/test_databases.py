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


import pytest

import os
import bob.bio.base
from bob.bio.base.test.utils import db_available
from bob.bio.base.test.test_database_implementations import check_database
import bob.core
from bob.extension.download import get_file
from bob.extension import rc

logger = bob.core.log.setup("bob.bio.face")


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
        logger.warn("Some annotations were None for {}".format(database_legacy.name))


@db_available("arface")
def test_arface():
    database = bob.bio.base.load_resource(
        "arface", "database", preferred_package="bob.bio.face"
    )
    try:
        check_database(database, groups=("dev", "eval"))
    except IOError as e:
        pytest.skip(
            "The database could not queried; probably the db.sql3 file is missing. Here is the error: '%s'"
            % e
        )
    try:
        _check_annotations(database)
    except IOError as e:
        pytest.skip(
            "The annotations could not be queried; probably the annotation files are missing. Here is the error: '%s'"
            % e
        )


@db_available("atnt")
def test_atnt():
    database = bob.bio.base.load_resource(
        "atnt", "database", preferred_package="bob.bio.face"
    )
    try:
        check_database(database)
    except IOError as e:
        pytest.skip(
            "The database could not queried; probably the db.sql3 file is missing. Here is the error: '%s'"
            % e
        )


@db_available("gbu")
def test_gbu():
    database = bob.bio.base.load_resource(
        "gbu", "database", preferred_package="bob.bio.face"
    )
    try:
        check_database(database, models_depend=True)
        check_database(database, protocol="Bad", models_depend=True)
        check_database(database, protocol="Ugly", models_depend=True)
    except IOError as e:
        pytest.skip(
            "The database could not queried; probably the db.sql3 file is missing. Here is the error: '%s'"
            % e
        )
    try:
        _check_annotations(database, limit_files=1000)
    except IOError as e:
        pytest.skip(
            "The annotations could not be queried; probably the annotation files are missing. Here is the error: '%s'"
            % e
        )


@db_available("lfw")
def test_lfw():
    database = bob.bio.base.load_resource(
        "lfw-restricted", "database", preferred_package="bob.bio.face"
    )
    try:
        check_database(database, training_depends=True, models_depend=True)
        check_database(
            database,
            groups=("dev", "eval"),
            protocol="fold1",
            training_depends=True,
            models_depend=True,
        )
        check_database(
            bob.bio.base.load_resource(
                "lfw-unrestricted", "database", preferred_package="bob.bio.face"
            ),
            training_depends=True,
            models_depend=True,
        )
    except IOError as e:
        pytest.skip(
            "The database could not queried; probably the db.sql3 file is missing. Here is the error: '%s'"
            % e
        )
    try:
        _check_annotations(database, limit_files=1000)
    except IOError as e:
        pytest.skip(
            "The annotations could not be queried; probably the annotation files are missing. Here is the error: '%s'"
            % e
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
    assert len(database.treferences()) == 8
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


@db_available("replay")
def test_replay_licit():
    database = bob.bio.base.load_resource(
        "replay-img-licit", "database", preferred_package="bob.bio.face"
    )
    try:
        check_database(database, groups=("dev", "eval"))
    except IOError as e:
        pytest.skip(
            "The database could not be queried; probably the db.sql3 file is missing. Here is the error: '%s'"
            % e
        )
    try:
        _check_annotations(database, topleft=True)
    except IOError as e:
        pytest.skip(
            "The annotations could not be queried; probably the annotation files are missing. Here is the error: '%s'"
            % e
        )


@db_available("replay")
def test_replay_spoof():
    database = bob.bio.base.load_resource(
        "replay-img-spoof", "database", preferred_package="bob.bio.face"
    )
    try:
        check_database(database, groups=("dev", "eval"))
    except IOError as e:
        pytest.skip(
            "The database could not be queried; probably the db.sql3 file is missing. Here is the error: '%s'"
            % e
        )
    try:
        _check_annotations(database, topleft=True)
    except IOError as e:
        pytest.skip(
            "The annotations could not be queried; probably the annotation files are missing. Here is the error: '%s'"
            % e
        )


def test_replaymobile():
    database = bob.bio.base.load_resource(
        "replaymobile-img", "database", preferred_package="bob.bio.face"
    )
    sample = database.probes()[0][0]
    assert hasattr(sample, "annotations")
    assert "reye" in sample.annotations
    assert "leye" in sample.annotations
    assert hasattr(sample, "path")
    assert hasattr(sample, "frame")
    assert len(database.references()) == 16
    assert len(database.references(group="eval")) == 12
    assert len(database.probes()) == 4160
    assert len(database.probes(group="eval")) == 3020
    # Only if data is available
    if rc.get("bob.db.replaymobile.directory", None):
        assert sample.annotations == {
            "bottomright": [734, 407],
            "topleft": [436, 182],
            "leye": [541, 350],
            "reye": [540, 245],
            "mouthleft": [655, 254],
            "mouthright": [657, 338],
            "nose": [591, 299],
        }
        assert sample.data.shape == (3, 1280, 720)
        assert sample.data[0, 0, 0] == 110


@pytest.mark.skipif(
    rc.get("bob.bio.face.ijbc.directory") is None,
    reason="IJBC original protocols not available. Please do `bob config set bob.bio.face.ijbc.directory [IJBC PATH]` to set the IJBC data path.",
)
def test_ijbc():
    from bob.bio.face.database import IJBCDatabase

    database = IJBCDatabase()

    # assert len(database.background_model_samples()) == 140732
    assert len(database.references()) == 3531
    assert len(database.probes()) == 19593
    num_comparisons = sum([len(item.references) for item in database.probes()])
    assert num_comparisons == 19557 + 15638932  # Genuine + Impostor


@db_available("fargo")
def test_fargo():
    database = bob.bio.base.load_resource(
        "fargo", "database", preferred_package="bob.bio.face"
    )
    try:
        check_database(database)
    except IOError as e:
        pytest.skip("The database could not queried; Here is the error: '%s'" % e)


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

    database = FRGCDatabase("2.0.1")

    assert len(database.background_model_samples()) == 12776
    assert len(database.references()) == 7572
    assert len(database.probes()) == 8456

    database = FRGCDatabase("2.0.2")
    assert len(database.background_model_samples()) == 12776
    assert len(database.references()) == 1893
    assert len(database.probes()) == 8456

    database = FRGCDatabase("2.0.4")
    assert len(database.background_model_samples()) == 12776
    assert len(database.references()) == 7572
    assert len(database.probes()) == 4228


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

