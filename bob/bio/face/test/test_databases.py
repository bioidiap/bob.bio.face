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


from nose.plugins.skip import SkipTest

import bob.bio.base
from bob.bio.base.test.utils import db_available
from bob.bio.base.test.test_database_implementations import check_database, check_database_zt


def _check_annotations(database):
    for file in database.all_files():
        annotations = database.annotations(file)
        if annotations is not None:
            assert isinstance(annotations, dict)
            assert 'reye' in annotations
            assert 'leye' in annotations


@db_available('arface')
def test_arface():
    database = bob.bio.base.load_resource('arface', 'database', preferred_package='bob.bio.face')
    try:
        check_database(database, groups=('dev', 'eval'))
        _check_annotations(database)
    except IOError as e:
        raise SkipTest(
            "The database could not queried; probably the db.sql3 file is missing. Here is the error: '%s'" % e)


@db_available('atnt')
def test_atnt():
    database = bob.bio.base.load_resource('atnt', 'database', preferred_package='bob.bio.face')
    try:
        check_database(database)
        _check_annotations(database)
    except IOError as e:
        raise SkipTest(
            "The database could not queried; probably the db.sql3 file is missing. Here is the error: '%s'" % e)


@db_available('banca')
def test_banca():
    database = bob.bio.base.load_resource('banca', 'database', preferred_package='bob.bio.face')
    try:
        check_database_zt(database)
        _check_annotations(database)
    except IOError as e:
        raise SkipTest(
            "The database could not be queried; probably the db.sql3 file is missing. Here is the error: '%s'" % e)


@db_available('caspeal')
def test_caspeal():
    database = bob.bio.base.load_resource('caspeal', 'database', preferred_package='bob.bio.face')
    try:
        check_database(database)
        check_database(database, protocol='aging')
        _check_annotations(database)
    except IOError as e:
        raise SkipTest(
            "The database could not queried; probably the db.sql3 file is missing. Here is the error: '%s'" % e)


#@db_available('frgc')
#def test_frgc():
#    import xml.sax
#    database = bob.bio.base.load_resource('frgc', 'database', preferred_package='bob.bio.face')
#    try:
#        _check_database(database, models_depend=True)
#        _check_database(database, protocol='2.0.2', models_depend=True)
#        _check_annotations(database)
#    except xml.sax.SAXException as e:
#        raise SkipTest(
#            "The database could not be opened, probably the original directory is wrong. Here is the error: '%s'" % e)

@db_available('cuhk_cufs')
def test_cuhk_cufs():
    database = bob.bio.base.load_resource('cuhk_cufs', 'database', preferred_package='bob.bio.face')
    try:
        check_database(database)
        _check_annotations(database)
    except IOError as e:
        raise SkipTest(
            "The database could not queried; probably the db.sql3 file is missing. Here is the error: '%s'" % e)


@db_available('gbu')
def test_gbu():
    database = bob.bio.base.load_resource('gbu', 'database', preferred_package='bob.bio.face')
    try:
        check_database(database, models_depend=True)
        check_database(database, protocol='Bad', models_depend=True)
        check_database(database, protocol = 'Ugly', models_depend=True)
        _check_annotations(database)
    except IOError as e:
        raise SkipTest(
            "The database could not queried; probably the db.sql3 file is missing. Here is the error: '%s'" % e)


@db_available('ijba')
def test_ijba():
    database = bob.bio.base.load_resource('ijba', 'database', preferred_package='bob.bio.face')
    try:
        check_database(database)
        _check_annotations(database)
    except IOError as e:
        raise SkipTest(
            "The database could not queried; probably the db.sql3 file is missing. Here is the error: '%s'" % e)


@db_available('lfw')
def test_lfw():
    database = bob.bio.base.load_resource('lfw-restricted', 'database', preferred_package='bob.bio.face')
    try:
        check_database(database, training_depends=True, models_depend=True)
        check_database(database, groups=('dev', 'eval'), protocol='fold1', training_depends=True, models_depend=True)
        check_database(bob.bio.base.load_resource('lfw-unrestricted', 'database', preferred_package='bob.bio.face'),
                       training_depends=True, models_depend=True)
        _check_annotations(database)
    except IOError as e:
        raise SkipTest(
            "The database could not queried; probably the db.sql3 file is missing. Here is the error: '%s'" % e)


@db_available('mobio')
def test_mobio():
    database = bob.bio.base.load_resource('mobio-image', 'database', preferred_package='bob.bio.face')
    try:
        check_database_zt(database, models_depend=True)
        check_database_zt(database, protocol='female', models_depend=True)
        check_database_zt(bob.bio.base.load_resource('mobio-male', 'database', preferred_package='bob.bio.face'),
                          models_depend=True)
        check_database_zt(bob.bio.base.load_resource('mobio-female', 'database', preferred_package='bob.bio.face'),
                          models_depend=True)
    except IOError as e:
        raise SkipTest(
            "The database could not be queried; probably the db.sql3 file is missing. Here is the error: '%s'" % e)

    try:
        _check_annotations(database)
    except Exception as e:
        raise SkipTest(
            "The annotations could not be queried; probably the annotation files are missing. Here is the error: '%s'" % e)


@db_available('multipie')
def test_multipie():
    database = bob.bio.base.load_resource('multipie', 'database', preferred_package='bob.bio.face')
    try:
        check_database_zt(database, training_depends=True)
        check_database_zt(bob.bio.base.load_resource('multipie-pose', 'database', preferred_package='bob.bio.face'),
                          training_depends=True)
    except IOError as e:
        raise SkipTest(
            "The database could not queried; probably the db.sql3 file is missing. Here is the error: '%s'" % e)
    except ValueError as e:
        raise SkipTest(
            "The database could not queried; probably the protocol is missing inside the db.sql3 file. Here is the error: '%s'" % e)

    try:
        _check_annotations(database)
    except Exception as e:
        raise SkipTest(
            "The annotations could not be queried; probably the annotation files are missing. Here is the error: '%s'" % e)


@db_available('scface')
def test_scface():
    database = bob.bio.base.load_resource('scface', 'database', preferred_package='bob.bio.face')
    try:
        check_database_zt(database)
        _check_annotations(database)
    except IOError as e:
        raise SkipTest(
            "The database could not be queried; probably the db.sql3 file is missing. Here is the error: '%s'" % e)


@db_available('xm2vts')
def test_xm2vts():
    database = bob.bio.base.load_resource('xm2vts', 'database', preferred_package='bob.bio.face')
    try:
        check_database(database, groups=('dev', 'eval'))
        check_database(database, groups=('dev', 'eval'), protocol='darkened-lp1')
        _check_annotations(database)
    except IOError as e:
        raise SkipTest(
            "The database could not be queried; probably the db.sql3 file is missing. Here is the error: '%s'" % e)


@db_available('replay')
def test_replay():
    database = bob.bio.base.load_resource('replay', 'database', preferred_package='bob.bio.face')
    try:
        check_database(database, groups=('dev', 'eval'))
        _check_annotations(database)
    except IOError as e:
        raise SkipTest(
            "The database could not be queried; probably the db.sql3 file is missing. Here is the error: '%s'" % e)
