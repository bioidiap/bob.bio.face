#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST
#
# Copyright (C) Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# This file contains the python (distutils/setuptools) instructions so your
# package can be installed on **any** host system. It defines some basic
# information like the package name for instance, or its homepage.
#
# It also defines which other packages this python package depends on and that
# are required for this package's operation. The python subsystem will make
# sure all dependent packages are installed or will install them for you upon
# the installation of this package.
#
# The 'buildout' system we use here will go further and wrap this package in
# such a way to create an isolated python working environment. Buildout will
# make sure that dependencies which are not yet installed do get installed, but
# **without** requiring administrative privileges on the host system. This
# allows you to test your package with new python dependencies w/o requiring
# administrative interventions.

from setuptools import setup, dist
dist.Distribution(dict(setup_requires=['bob.extension']))

from bob.extension.utils import load_requirements, find_packages
install_requires = load_requirements()

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    # This is the basic information about your project. Modify all this
    # information before releasing code publicly.
    name = 'bob.bio.face',
    version = open("version.txt").read().rstrip(),
    description = 'Tools for running face recognition experiments',

    url = 'https://www.github.com/bioidiap/bob.bio.face',
    license = 'GPLv3',
    author = '<YourName>',
    author_email = '<YourEmail>',
    keywords = 'bob, biometric recognition, evaluation',

    # If you have a better, long description of your package, place it on the
    # 'doc' directory and then hook it here
    long_description = open('README.rst').read(),

    # This line is required for any distutils based packaging.
    packages = find_packages(),
    include_package_data = True,
    zip_safe=False,

    # This line defines which packages should be installed when you "install"
    # this package. All packages that are mentioned here, but are not installed
    # on the current system will be installed locally and only visible to the
    # scripts of this package. Don't worry - You won't need administrative
    # privileges when using buildout.
    install_requires = install_requires,

    # Your project should be called something like 'bob.<foo>' or
    # 'bob.<foo>.<bar>'. To implement this correctly and still get all your
    # packages to be imported w/o problems, you need to implement namespaces
    # on the various levels of the package and declare them here. See more
    # about this here:
    # http://peak.telecommunity.com/DevCenter/setuptools#namespace-packages
    #
    # Our database packages are good examples of namespace implementations
    # using several layers. You can check them out here:
    # https://github.com/idiap/bob/wiki/Satellite-Packages


    # This entry defines which scripts you will have inside the 'bin' directory
    # once you install the package (or run 'bin/buildout'). The order of each
    # entry under 'console_scripts' is like this:
    #   script-name-at-bin-directory = module.at.your.library:function
    #
    # The module.at.your.library is the python file within your library, using
    # the python syntax for directories (i.e., a '.' instead of '/' or '\').
    # This syntax also omits the '.py' extension of the filename. So, a file
    # installed under 'example/foo.py' that contains a function which
    # implements the 'main()' function of particular script you want to have
    # should be referred as 'example.foo:main'.
    #
    # In this simple example we will create a single program that will print
    # the version of bob.
    entry_points = {

      # scripts should be declared using this entry:
      'console_scripts' : [
        'baselines.py      = bob.bio.face.script.baselines:main',
        'display_face_annotations.py = bob.bio.face.script.display_face_annotations:main'
      ],

      'bob.bio.database': [
        'atnt              = bob.bio.face.config.database.atnt:database',
        'arface            = bob.bio.face.config.database.arface:database',
        'banca             = bob.bio.face.config.database.banca_english:database',
        'caspeal           = bob.bio.face.config.database.caspeal:database',
        'frgc              = bob.bio.face.config.database.frgc:database',
        'gbu               = bob.bio.face.config.database.gbu:database',
        'lfw-restricted    = bob.bio.face.config.database.lfw_restricted:database',
        'lfw-unrestricted  = bob.bio.face.config.database.lfw_unrestricted:database',
        'mobio-image       = bob.bio.face.config.database.mobio_image:database',
        'mobio-male        = bob.bio.face.config.database.mobio_male:database', # MOBIO gender-dependent training
        'mobio-female      = bob.bio.face.config.database.mobio_female:database', # MOBIO gender-dependent training
        'multipie          = bob.bio.face.config.database.multipie:database',
        'multipie-pose     = bob.bio.face.config.database.multipie_pose:database',
        'scface            = bob.bio.face.config.database.scface:database',
        'xm2vts            = bob.bio.face.config.database.xm2vts:database',
      ],

      'bob.bio.preprocessor': [
        'base              = bob.bio.face.config.preprocessor.base:preprocessor', # simple color conversion
        'face-crop-eyes    = bob.bio.face.config.preprocessor.face_crop_eyes:preprocessor', # face crop
        'landmark-detect   = bob.bio.face.config.preprocessor.face_detect:preprocessor', # face detection + landmark detection + cropping
        'face-detect       = bob.bio.face.config.preprocessor.face_detect:preprocessor_no_eyes', # face detection + cropping

        'inorm-lbp-crop    = bob.bio.face.config.preprocessor.inorm_lbp:preprocessor', # face crop + inorm-lbp
        'tan-triggs-crop   = bob.bio.face.config.preprocessor.tan_triggs:preprocessor', # face crop + Tan&Triggs
        'histogram-crop    = bob.bio.face.config.preprocessor.histogram_equalization:preprocessor', # face crop + histogram equalization
        'self-quotient-crop= bob.bio.face.config.preprocessor.self_quotient_image:preprocessor', # face crop + self quotient image

        'inorm-lbp-landmark = bob.bio.face.config.preprocessor.inorm_lbp:preprocessor_landmark', # face detection + landmark detection + cropping + inorm-lbp
        'tan-triggs-landmark = bob.bio.face.config.preprocessor.tan_triggs:preprocessor_landmark', # face detection + landmark detection + cropping + Tan&Triggs
        'histogram-landmark = bob.bio.face.config.preprocessor.histogram_equalization:preprocessor_landmark', # face detection + landmark detection + cropping + histogram equalization
        'self-quotient-landmark = bob.bio.face.config.preprocessor.self_quotient_image:preprocessor_landmark', # face detection + landmark detection + cropping + self quotient image

        'inorm-lbp         = bob.bio.face.config.preprocessor.inorm_lbp:preprocessor_no_crop', # inorm-lbp w/o face-crop
        'tan-triggs        = bob.bio.face.config.preprocessor.tan_triggs:preprocessor_no_crop', # Tan&Triggs w/o face-crop
        'histogram         = bob.bio.face.config.preprocessor.histogram_equalization:preprocessor_no_crop', # histogram equalization w/o face-crop
        'self-quotient     = bob.bio.face.config.preprocessor.self_quotient_image:preprocessor_no_crop', # self quotient image w/o face-crop
      ],

      'bob.bio.extractor': [
        'dct-blocks        = bob.bio.face.config.extractor.dct_blocks:extractor', # DCT blocks
        'grid-graph        = bob.bio.face.config.extractor.grid_graph:extractor', # Grid graph
        'lgbphs            = bob.bio.face.config.extractor.lgbphs:extractor', # LGBPHS
        'eigenface         = bob.bio.face.config.extractor.eigenface:extractor', # Eigenface
      ],

      'bob.bio.algorithm': [
        'gabor-jet         = bob.bio.face.config.algorithm.gabor_jet:algorithm', # Gabor jet comparison
        'histogram         = bob.bio.face.config.algorithm.histogram:algorithm', # LGBPHS histograms
        'bic-jets          = bob.bio.face.config.algorithm.bic_jets:algorithm', # BIC on gabor jets
      ],
   },

    # Classifiers are important if you plan to distribute this package through
    # PyPI. You can find the complete list of classifiers that are valid and
    # useful here (http://pypi.python.org/pypi?%3Aaction=list_classifiers).
    classifiers = [
      'Framework :: Bob',
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
