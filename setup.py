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

dist.Distribution(dict(setup_requires=["bob.extension"]))

from bob.extension.utils import load_requirements, find_packages

install_requires = load_requirements()

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(
    # This is the basic information about your project. Modify all this
    # information before releasing code publicly.
    name="bob.bio.face",
    version=open("version.txt").read().rstrip(),
    description="Tools for running face recognition experiments",
    url="https://gitlab.idiap.ch/bob/bob.bio.face",
    license="BSD",
    author="Manuel Gunther",
    author_email="siebenkopf@googlemail.com",
    keywords="bob, biometric recognition, evaluation",
    # If you have a better, long description of your package, place it on the
    # 'doc' directory and then hook it here
    long_description=open("README.rst").read(),
    # This line is required for any distutils based packaging.
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    # This line defines which packages should be installed when you "install"
    # this package. All packages that are mentioned here, but are not installed
    # on the current system will be installed locally and only visible to the
    # scripts of this package. Don't worry - You won't need administrative
    # privileges when using buildout.
    install_requires=install_requires,
    # Your project should be called something like 'bob.<foo>' or
    # 'bob.<foo>.<bar>'. To implement this correctly and still get all your
    # packages to be imported w/o problems, you need to implement namespaces
    # on the various levels of the package and declare them here. See more
    # about this here:
    # http://peak.telecommunity.com/DevCenter/setuptools#namespace-packages
    #
    # Our database packages are good examples of namespace implementations
    # using several layers. You can check them out here:
    # https://www.idiap.ch/software/bob/packages
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
    entry_points={
        # scripts should be declared using this entry:
        "console_scripts": [],
        "bob.bio.database": [
            "arface            = bob.bio.face.config.database.arface:database",
            "atnt              = bob.bio.face.config.database.atnt:database",
            "gbu               = bob.bio.face.config.database.gbu:database",
            "ijbc-11              = bob.bio.face.config.database.ijbc:database",
            "lfw-restricted    = bob.bio.face.config.database.lfw_restricted:database",
            "lfw-unrestricted  = bob.bio.face.config.database.lfw_unrestricted:database",
            "mobio-male       = bob.bio.face.config.database.mobio_male:database",
            "mobio-all        = bob.bio.face.config.database.mobio_all:database",
            "multipie          = bob.bio.face.config.database.multipie:database",
            "multipie-pose     = bob.bio.face.config.database.multipie_pose:database",
            "replay-img-licit  = bob.bio.face.config.database.replay:replay_licit",
            "replay-img-spoof  = bob.bio.face.config.database.replay:replay_spoof",
            "replaymobile-img-licit  = bob.bio.face.config.database.replaymobile:replaymobile_licit",
            "replaymobile-img-spoof  = bob.bio.face.config.database.replaymobile:replaymobile_spoof",
            "fargo  = bob.bio.face.config.database.fargo:database",
            "meds = bob.bio.face.config.database.meds:database",
            "morph = bob.bio.face.config.database.morph:database",
            "casia-africa = bob.bio.face.config.database.casia_africa:database",
            "pola-thermal = bob.bio.face.config.database.pola_thermal:database",
            "cbsr-nir-vis-2 = bob.bio.face.config.database.cbsr_nir_vis_2:database",
        ],
        "bob.bio.annotator": [
            "facedetect               = bob.bio.face.config.annotator.facedetect:annotator",
            "facedetect-eye-estimate  = bob.bio.face.config.annotator.facedetect_eye_estimate:annotator",
            "flandmark                = bob.bio.face.config.annotator.flandmark:annotator",
            "mtcnn                    = bob.bio.face.config.annotator.mtcnn:annotator",
        ],
        "bob.bio.transformer": [
            "facedetect-eye-estimate = bob.bio.face.config.annotator.facedetect_eye_estimate:transformer",
            "facedetect = bob.bio.face.config.annotator.facedetect:transformer",
            "flandmark = bob.bio.face.config.annotator.flandmark:annotator",
            "mtcnn = bob.bio.face.config.annotator.mtcnn:transformer",
            "facenet-sanderberg = bob.bio.face.config.baseline.facenet_sanderberg:transformer",
            "inception-resnetv1-casiawebface = bob.bio.face.config.baseline.inception_resnetv1_casiawebface:transformer",
            "inception-resnetv2-casiawebface = bob.bio.face.config.baseline.inception_resnetv2_casiawebface:transformer",
            "inception-resnetv1-msceleb = bob.bio.face.config.baseline.inception_resnetv1_msceleb:transformer",
            "inception-resnetv2-msceleb = bob.bio.face.config.baseline.inception_resnetv2_msceleb:transformer",
            "arcface-insightface = bob.bio.face.config.baseline.arcface_insightface:transformer",
            "gabor-graph = bob.bio.face.config.baseline.gabor_graph:transformer",
            "lgbphs = bob.bio.face.config.baseline.lgbphs:transformer",
            "dummy = bob.bio.face.config.baseline.dummy:transformer",
        ],
        # baselines
        "bob.bio.pipeline": [
            "facenet-sanderberg = bob.bio.face.config.baseline.facenet_sanderberg:pipeline",
            "inception-resnetv1-casiawebface = bob.bio.face.config.baseline.inception_resnetv1_casiawebface:pipeline",
            "inception-resnetv2-casiawebface = bob.bio.face.config.baseline.inception_resnetv2_casiawebface:pipeline",
            "inception-resnetv1-msceleb = bob.bio.face.config.baseline.inception_resnetv1_msceleb:pipeline",
            "inception-resnetv2-msceleb = bob.bio.face.config.baseline.inception_resnetv2_msceleb:pipeline",
            "gabor_graph = bob.bio.face.config.baseline.gabor_graph:pipeline",
            "arcface-insightface = bob.bio.face.config.baseline.arcface_insightface:pipeline",
            "lgbphs = bob.bio.face.config.baseline.lgbphs:pipeline",
            "lda = bob.bio.face.config.baseline.lda:pipeline",
            "dummy = bob.bio.face.config.baseline.dummy:pipeline",
            "resnet50-msceleb-arcface-2021 = bob.bio.face.config.baseline.resnet50_msceleb_arcface_2021:pipeline",
            "resnet50-vgg2-arcface-2021 = bob.bio.face.config.baseline.resnet50_vgg2_arcface_2021:pipeline",
            "mobilenetv2-msceleb-arcface-2021 = bob.bio.face.config.baseline.mobilenetv2_msceleb_arcface_2021",
        ],
        "bob.bio.config": [
            "facenet-sanderberg = bob.bio.face.config.baseline.facenet_sanderberg",
            "inception-resnetv1-casiawebface = bob.bio.face.config.baseline.inception_resnetv1_casiawebface",
            "inception-resnetv2-casiawebface = bob.bio.face.config.baseline.inception_resnetv2_casiawebface",
            "inception-resnetv1-msceleb = bob.bio.face.config.baseline.inception_resnetv1_msceleb",
            "inception-resnetv2-msceleb = bob.bio.face.config.baseline.inception_resnetv2_msceleb",
            "gabor_graph = bob.bio.face.config.baseline.gabor_graph",
            "arcface-insightface = bob.bio.face.config.baseline.arcface_insightface",
            "lgbphs = bob.bio.face.config.baseline.lgbphs",
            "lda = bob.bio.face.config.baseline.lda",
            "arface            = bob.bio.face.config.database.arface",
            "atnt              = bob.bio.face.config.database.atnt",
            "gbu               = bob.bio.face.config.database.gbu",
            "ijbc-11              = bob.bio.face.config.database.ijbc",
            "lfw-restricted    = bob.bio.face.config.database.lfw_restricted",
            "lfw-unrestricted  = bob.bio.face.config.database.lfw_unrestricted",
            "mobio-male       = bob.bio.face.config.database.mobio_male",
            "mobio-all        = bob.bio.face.config.database.mobio_all",
            "multipie          = bob.bio.face.config.database.multipie",
            "multipie-pose     = bob.bio.face.config.database.multipie_pose",
            "replay-img-licit  = bob.bio.face.config.database.replay_licit",
            "replay-img-spoof  = bob.bio.face.config.database.replay_spoof",
            "replaymobile-img-licit  = bob.bio.face.config.database.replaymobile_licit",
            "replaymobile-img-spoof  = bob.bio.face.config.database.replaymobile_spoof",
            "fargo  = bob.bio.face.config.database.fargo",
            "meds = bob.bio.face.config.database.meds",
            "morph = bob.bio.face.config.database.morph",
            "casia-africa = bob.bio.face.config.database.casia_africa",
            "pola-thermal = bob.bio.face.config.database.pola_thermal",
            "cbsr-nir-vis-2 = bob.bio.face.config.database.cbsr_nir_vis_2",
            "resnet50-msceleb-arcface-2021 = bob.bio.face.config.baseline.resnet50_msceleb_arcface_2021",
            "resnet50-vgg2-arcface-2021 = bob.bio.face.config.baseline.resnet50_vgg2_arcface_2021",
            "mobilenetv2-msceleb-arcface-2021 = bob.bio.face.config.baseline.mobilenetv2_msceleb_arcface_2021",
        ],
        "bob.bio.cli": [
            "display-face-annotations          = bob.bio.face.script.display_face_annotations:display_face_annotations",
        ],
    },
    # Classifiers are important if you plan to distribute this package through
    # PyPI. You can find the complete list of classifiers that are valid and
    # useful here (http://pypi.python.org/pypi?%3Aaction=list_classifiers).
    classifiers=[
        "Framework :: Bob",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
