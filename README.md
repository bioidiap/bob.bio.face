[![badge doc](https://img.shields.io/badge/docs-v8.0.0-orange.svg)](https://www.idiap.ch/software/bob/docs/bob/bob.bio.face/v8.0.0/sphinx/index.html)
[![badge pipeline](https://gitlab.idiap.ch/bob/bob.bio.face/badges/v8.0.0/pipeline.svg)](https://gitlab.idiap.ch/bob/bob.bio.face/commits/v8.0.0)
[![badge coverage](https://gitlab.idiap.ch/bob/bob.bio.face/badges/v8.0.0/coverage.svg)](https://www.idiap.ch/software/bob/docs/bob/bob.bio.face/v8.0.0/coverage/)
[![badge gitlab](https://img.shields.io/badge/gitlab-project-0000c0.svg)](https://gitlab.idiap.ch/bob/bob.bio.face)

# Run face recognition algorithms

This package is part of the signal-processing and machine learning toolbox
[Bob](https://www.idiap.ch/software/bob).
This package is part of the `bob.bio` packages, which allow to run
comparable and reproducible biometric recognition experiments on publicly
available datasets.

This package contains functionality to run face recognition experiments.
It is an extension to the
[bob.bio.base](https://pypi.python.org/pypi/bob.bio.base) package, which
provides the basic scripts.
In this package, utilities that are specific for face recognition are
contained, such as:

* Image databases
* Image preprocesors, including face detection and facial image alignment
* Image feature extractors
* Recognition algorithms based on image features

## Installation

Complete Bob's
[installation instructions](https://www.idiap.ch/software/bob/install). Then,
to install this package, run:

``` sh
conda install bob.bio.face
```

## Contact

For questions or reporting issues to this software package, contact our
development [mailing list](https://www.idiap.ch/software/bob/discuss).
