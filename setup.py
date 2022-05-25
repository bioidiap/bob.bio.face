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

from setuptools import dist, setup

dist.Distribution(dict(setup_requires=["bob.extension"]))

from bob.extension.utils import find_packages, load_requirements

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
            "arface-all = bob.bio.face.config.database.arface_all:database",
            "arface-expression = bob.bio.face.config.database.arface_expression:database",
            "arface-illumination = bob.bio.face.config.database.arface_illumination:database",
            "arface-occlusion = bob.bio.face.config.database.arface_occlusion:database",
            "arface-occlusion-and-illumination = bob.bio.face.config.database.arface_occlusion_and_illumination:database",
            "casia-africa             = bob.bio.face.config.database.casia_africa:database",
            "caspeal-accessory = bob.bio.face.config.database.caspeal_accessory:database",
            "caspeal-aging = bob.bio.face.config.database.caspeal_aging:database",
            "caspeal-background = bob.bio.face.config.database.caspeal_background:database",
            "caspeal-distance = bob.bio.face.config.database.caspeal_distance:database",
            "caspeal-expression = bob.bio.face.config.database.caspeal_expression:database",
            "caspeal-lighting = bob.bio.face.config.database.caspeal_lighting:database",
            "frgc-exp1                = bob.bio.face.config.database.frgc_experiment1:database",
            "frgc-exp2                = bob.bio.face.config.database.frgc_experiment2:database",
            "frgc-exp4                = bob.bio.face.config.database.frgc_experiment4:database",
            "gbu-good                      = bob.bio.face.config.database.gbu_good:database",
            "gbu-bad                      = bob.bio.face.config.database.gbu_bad:database",
            "gbu-ugly                      = bob.bio.face.config.database.gbu_ugly:database",
            "ijbc-test1                            = bob.bio.face.config.database.ijbc_test1:database",
            "ijbc-test2                            = bob.bio.face.config.database.ijbc_test2:database",
            "ijbc-test4-g1                            = bob.bio.face.config.database.ijbc_test4_g1:database",
            "ijbc-test4-g2                            = bob.bio.face.config.database.ijbc_test4_g2:database",
            "lfw-o1           = bob.bio.face.config.database.lfw_o1:database",
            "lfw-o2           = bob.bio.face.config.database.lfw_o2:database",
            "lfw-o3           = bob.bio.face.config.database.lfw_o3:database",
            "lfw-view2           = bob.bio.face.config.database.lfw_view2:database",
            "meds                     = bob.bio.face.config.database.meds:database",
            "mobio-all                = bob.bio.face.config.database.mobio_all:database",
            "mobio-male               = bob.bio.face.config.database.mobio_male:database",
            "morph                    = bob.bio.face.config.database.morph:database",
            "multipie                 = bob.bio.face.config.database.multipie:database",
            "multipie-pose            = bob.bio.face.config.database.multipie_pose:database",
            "pola-thermal             = bob.bio.face.config.database.pola_thermal:database",
            "replaymobile-img         = bob.bio.face.config.database.replaymobile:database",
            "rfw-original             = bob.bio.face.config.database.rfw_original:database",
            "rfw-idiap                = bob.bio.face.config.database.rfw_idiap:database",
            "scface                   = bob.bio.face.config.database.scface_combined:database",
            "scface-close             = bob.bio.face.config.database.scface_close:database",
            "scface-medium            = bob.bio.face.config.database.scface_medium:database",
            "scface-far               = bob.bio.face.config.database.scface_far:database",
            "scface-ir                = bob.bio.face.config.database.scface_ir:database",
            "vgg2-full                = bob.bio.face.config.database.vgg2_full:database",
            "vgg2-short                = bob.bio.face.config.database.vgg2_short:database",
        ],
        "bob.bio.annotator": [
            "mtcnn                    = bob.bio.face.config.annotator.mtcnn:annotator",
            "tinyface                 = bob.bio.face.config.annotator.tinyface:annotator",
        ],
        # baselines
        "bob.bio.pipeline": [
            "afffe                                 = bob.bio.face.config.baseline.afffe:pipeline",
            "arcface-insightface                   = bob.bio.face.config.baseline.arcface_insightface:pipeline",
            "dummy                                 = bob.bio.face.config.baseline.dummy:pipeline",
            "facenet-sanderberg                    = bob.bio.face.config.baseline.facenet_sanderberg:pipeline",
            "inception-resnetv1-casiawebface       = bob.bio.face.config.baseline.inception_resnetv1_casiawebface:pipeline",
            "inception-resnetv1-msceleb            = bob.bio.face.config.baseline.inception_resnetv1_msceleb:pipeline",
            "inception-resnetv2-casiawebface       = bob.bio.face.config.baseline.inception_resnetv2_casiawebface:pipeline",
            "inception-resnetv2-msceleb            = bob.bio.face.config.baseline.inception_resnetv2_msceleb:pipeline",
            "iresnet100                            = bob.bio.face.config.baseline.iresnet100:pipeline",
            "iresnet100-elastic                    = bob.bio.face.config.baseline.iresnet100_elastic:pipeline",
            "iresnet100-msceleb-idiap-20210623     = bob.bio.face.config.baseline.iresnet100_msceleb_arcface_20210623:pipeline",
            "iresnet34                             = bob.bio.face.config.baseline.iresnet34:pipeline",
            "iresnet50                             = bob.bio.face.config.baseline.iresnet50:pipeline",
            "iresnet50-msceleb-idiap-20210623      = bob.bio.face.config.baseline.iresnet50_msceleb_arcface_20210623:pipeline",
            "mobilenetv2-msceleb-arcface-2021      = bob.bio.face.config.baseline.mobilenetv2_msceleb_arcface_2021:pipeline",
            "resnet50-msceleb-arcface-2021         = bob.bio.face.config.baseline.resnet50_msceleb_arcface_2021:pipeline",
            "resnet50-msceleb-arcface-20210521     = bob.bio.face.config.baseline.resnet50_msceleb_arcface_20210521:pipeline",
            "resnet101-msceleb-arcface-20210521    = bob.bio.face.config.baseline.resnet101_msceleb_arcface_20210521:pipeline",
            "resnet50-vgg2-arcface-2021            = bob.bio.face.config.baseline.resnet50_vgg2_arcface_2021:pipeline",
            "vgg16-oxford                          = bob.bio.face.config.baseline.vgg16_oxford:pipeline",
            "attentionnet                          = bob.bio.face.config.baseline.attention_net:pipeline",
            "resnest                               = bob.bio.face.config.baseline.resne_st:pipeline",
            "mobilefacenet                         = bob.bio.face.config.baseline.mobile_facenet:pipeline",
            "resnet                                = bob.bio.face.config.baseline.resnet:pipeline",
            "efficientnet                          = bob.bio.face.config.baseline.efficient_net:pipeline",
            "tfnas                                 = bob.bio.face.config.baseline.tf_nas:pipeline",
            "hrnet                                 = bob.bio.face.config.baseline.hr_net:pipeline",
            "rexnet                                = bob.bio.face.config.baseline.rex_net:pipeline",
            "ghostnet                              = bob.bio.face.config.baseline.ghost_net:pipeline",
            "oxford-resnet50-scratch-dag-vgg2              = bob.bio.face.config.baseline.oxford_resnet50_scratch_dag_vgg2:pipeline",
            "oxford-resnet50-ft-dag-vgg2              = bob.bio.face.config.baseline.oxford_resnet50_ft_dag_vgg2:pipeline",
            "oxford-senet50-scratch-dag-vgg2              = bob.bio.face.config.baseline.oxford_senet50_scratch_dag_vgg2:pipeline",
            "oxford-senet50-ft-dag-vgg2              = bob.bio.face.config.baseline.oxford_senet50_ft_dag_vgg2:pipeline",
        ],
        "bob.bio.config": [
            # pipelines
            "afffe                                 = bob.bio.face.config.baseline.afffe",
            "arcface-insightface                   = bob.bio.face.config.baseline.arcface_insightface",
            "facenet-sanderberg                    = bob.bio.face.config.baseline.facenet_sanderberg",
            "inception-resnetv1-casiawebface       = bob.bio.face.config.baseline.inception_resnetv1_casiawebface",
            "inception-resnetv1-msceleb            = bob.bio.face.config.baseline.inception_resnetv1_msceleb",
            "inception-resnetv2-casiawebface       = bob.bio.face.config.baseline.inception_resnetv2_casiawebface",
            "inception-resnetv2-msceleb            = bob.bio.face.config.baseline.inception_resnetv2_msceleb",
            "iresnet100                            = bob.bio.face.config.baseline.iresnet100",
            "iresnet100-msceleb-idiap-20210623     = bob.bio.face.config.baseline.iresnet100_msceleb_arcface_20210623",
            "iresnet34                             = bob.bio.face.config.baseline.iresnet34",
            "iresnet50                             = bob.bio.face.config.baseline.iresnet50",
            "iresnet50-msceleb-idiap-20210623      = bob.bio.face.config.baseline.iresnet50_msceleb_arcface_20210623",
            "iresnet100-elastic                    = bob.bio.face.config.baseline.iresnet100_elastic",
            "mobilenetv2-msceleb-arcface-2021      = bob.bio.face.config.baseline.mobilenetv2_msceleb_arcface_2021",
            "resnet50-msceleb-arcface-2021         = bob.bio.face.config.baseline.resnet50_msceleb_arcface_2021",
            "resnet50-msceleb-arcface-20210521     = bob.bio.face.config.baseline.resnet50_msceleb_arcface_20210521",
            "resnet101-msceleb-arcface-20210521    = bob.bio.face.config.baseline.resnet101_msceleb_arcface_20210521",
            "resnet50-vgg2-arcface-2021            = bob.bio.face.config.baseline.resnet50_vgg2_arcface_2021",
            "vgg16-oxford                          = bob.bio.face.config.baseline.vgg16_oxford",
            "attentionnet                          = bob.bio.face.config.baseline.attention_net",
            "resnest                               = bob.bio.face.config.baseline.resne_st",
            "mobilefacenet                         = bob.bio.face.config.baseline.mobile_facenet",
            "resnet                                = bob.bio.face.config.baseline.resnet",
            "efficientnet                          = bob.bio.face.config.baseline.efficient_net",
            "tfnas                                 = bob.bio.face.config.baseline.tf_nas",
            "hrnet                                 = bob.bio.face.config.baseline.hr_net",
            "rexnet                                = bob.bio.face.config.baseline.rex_net",
            "ghostnet                              = bob.bio.face.config.baseline.ghost_net",
            # databases
            "casia-africa                          = bob.bio.face.config.database.casia_africa",
            "frgc-exp1                             = bob.bio.face.config.database.frgc_experiment1",
            "frgc-exp2                             = bob.bio.face.config.database.frgc_experiment2",
            "frgc-exp4                             = bob.bio.face.config.database.frgc_experiment4",
            "gbu-good                              = bob.bio.face.config.database.gbu_good",
            "gbu-bad                               = bob.bio.face.config.database.gbu_bad",
            "gbu-ugly                              = bob.bio.face.config.database.gbu_ugly",
            "ijbc-test1                            = bob.bio.face.config.database.ijbc_test1",
            "ijbc-test2                            = bob.bio.face.config.database.ijbc_test2",
            "ijbc-test4-g1                         = bob.bio.face.config.database.ijbc_test4_g1",
            "ijbc-test4-g2                         = bob.bio.face.config.database.ijbc_test4_g2",
            "lfw-o1                                = bob.bio.face.config.database.lfw_o1",
            "lfw-o2                                = bob.bio.face.config.database.lfw_o2",
            "lfw-o3                                = bob.bio.face.config.database.lfw_o3",
            "lfw-view2                             = bob.bio.face.config.database.lfw_view2",
            "meds                                  = bob.bio.face.config.database.meds",
            "mobio-all                             = bob.bio.face.config.database.mobio_all",
            "mobio-male                            = bob.bio.face.config.database.mobio_male",
            "morph                                 = bob.bio.face.config.database.morph",
            "multipie                              = bob.bio.face.config.database.multipie",
            "multipie-pose                         = bob.bio.face.config.database.multipie_pose",
            "pola-thermal                          = bob.bio.face.config.database.pola_thermal",
            "replaymobile-img                      = bob.bio.face.config.database.replaymobile",
            "rfw-original                          = bob.bio.face.config.database.rfw_original",
            "rfw-idiap                             = bob.bio.face.config.database.rfw_idiap",
            "scface                                = bob.bio.face.config.database.scface_combined",
            "scface-close                          = bob.bio.face.config.database.scface_close",
            "scface-medium                         = bob.bio.face.config.database.scface_medium",
            "scface-far                            = bob.bio.face.config.database.scface_far",
            "scface-ir                             = bob.bio.face.config.database.scface_ir",
            "arface-all                            = bob.bio.face.config.database.arface_all",
            "arface-expression                     = bob.bio.face.config.database.arface_expression",
            "arface-illumination                   = bob.bio.face.config.database.arface_illumination",
            "arface-occlusion                      = bob.bio.face.config.database.arface_occlusion",
            "arface-occlusion-and-illumination     = bob.bio.face.config.database.arface_occlusion_and_illumination",
            "caspeal-accessory                     = bob.bio.face.config.database.caspeal_accessory",
            "caspeal-aging                         = bob.bio.face.config.database.caspeal_aging",
            "caspeal-background                    = bob.bio.face.config.database.caspeal_background",
            "caspeal-distance                      = bob.bio.face.config.database.caspeal_distance",
            "caspeal-expression                    = bob.bio.face.config.database.caspeal_expression",
            "caspeal-lighting                      = bob.bio.face.config.database.caspeal_lighting",
            "oxford-resnet50-scratch-dag-vgg2      = bob.bio.face.config.baseline.oxford_resnet50_scratch_dag_vgg2",
            "oxford-resnet50-ft-dag-vgg2           = bob.bio.face.config.baseline.oxford_resnet50_ft_dag_vgg2",
            "oxford-senet50-scratch-dag-vgg2       = bob.bio.face.config.baseline.oxford_senet50_scratch_dag_vgg2",
            "oxford-senet50-ft-dag-vgg2            = bob.bio.face.config.baseline.oxford_senet50_ft_dag_vgg2",
            "vgg2-short                            = bob.bio.face.config.database.vgg2_short",
            "vgg2-full                            = bob.bio.face.config.database.vgg2_full",
            "vgg2-short-with-eval                  = bob.bio.face.config.database.vgg2_short_with_eval",
            "vgg2-full-with-eval                  = bob.bio.face.config.database.vgg2_full_with_eval",
        ],
        "bob.bio.cli": [
            "display-face-annotations          = bob.bio.face.script.display_face_annotations:display_face_annotations",
            "face                              = bob.bio.face.script.face:face",
        ],
        "bob.bio.face.cli": [
            "plots                     = bob.bio.face.script.plots:plots",
        ],
        "bob.bio.face.plots.cli": [
            "multipie-pose                     = bob.bio.face.script.face_plots_commands:multipie_pose",
            "multipie-expression               = bob.bio.face.script.face_plots_commands:multipie_expression",
            "scface                            = bob.bio.face.script.face_plots_commands:scface_distance",
            "arface                            = bob.bio.face.script.face_plots_commands:arface",
            "mobio-gender                      = bob.bio.face.script.face_plots_commands:mobio_gender",
            "gbu                               = bob.bio.face.script.face_plots_commands:gbu",
            "ijbc                              = bob.bio.face.script.face_plots_commands:ijbc",
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
