
======================
Implementation Details
======================

Image preprocessing
-------------------

Image preprocessing is an important stage for face recognition.
In :ref:`bob.bio.face <bob.bio.face>`, several different algorithms to perform photometric enhancement of facial images are implemented.
These algorithms rely on facial images, which are aligned according to the eye locations, and scaled to a specific image resolution.

Face cropping
~~~~~~~~~~~~~

However, for most of the image databases, in the original face images the faces are not aligned, but instead the eye locations are labeled by hand.
Hence, before the photometric enhancement algorithms can be applied, faces must be aligned according to the hand-labeled eye locations.
This can be achieved using the :py:class:`bob.bio.face.preprocessor.FaceCrop` class.
It will take the image and the hand-labeled eye locations and crop the face according to some parameters, which can be defined in its constructor.

So, now we have a preprocessors to perform face cropping, and some preprocessors to perform photometric enhancement.
However, we might want to have a photometric enhancement *on top of* the aligned faces.
In theory, there are several ways to achieve this:

1. Copy the face alignment code into all photometric enhancement classes.

   As copying code is generally a bad choice, we drop this option.


2. Use the face cropping as a base class and derive the photometric enhancement classes from it.

   This option is worth implementing, and this was the way, the FaceRecLib_ handled preprocessing.
   However, it required to copy code inside the configuration files.
   This means that, when we want to run on a different image resolution, we need to change all configuration files.
   Option 2 dropped.


3. Provide the face cropper as parameter to the photometric enhancement classes.

   This option has the advantage that the configuration has to be written only once.
   Also, we might change the face cropper to something else later, without needing to the the photometric enhancement code later on.
   Option 3 accepted.

Now, we have a closer look into how the image preprocessing is implemented.
Let's take the example of the :py:class:`bob.bio.face.preprocessor.TanTriggs`.
The constructor takes a ``face_cropper`` as parameter.
This ``face_cropper`` can be ``None``, when the images are already aligned.
It can also be a :py:class:`bob.bio.face.preprocessor.FaceCrop` object, which is contains the information, how faces are cropped.
The :py:class:`bob.bio.face.preprocessor.TanTriggs` algorithm will use the ``face_cropper`` to crop the face, by passing the image and the annotations to the :py:meth:`bob.bio.face.preprocessor.FaceCrop.crop_face` function, perform the photometric enhancement on the cropped image, and return the result.

So far, there is no advantage of option 2 over option 3, since the parameters for face cropping still have to be specified in the configuration file.
But now comes the clue: The third option, how a ``face_cropper`` can be passed to the constructor is as a :ref:`Resource <bob.bio.face.preprocessors>` key, such as ``'face-crop-eyes'``.
This will load the face cropping configuration from the registered resource, which has to be generated only once.
So, to generate a TanTriggs preprocessor that performs face cropping, you can create:

.. code-block:: py

   preprocessor = bob.bio.face.preprocessor.TanTriggs(face_cropper = 'face-crop-eyes')


Face detection
~~~~~~~~~~~~~~

Alright.
Now if you have swallowed that, there comes the next step: face detection.
Some of the databases do neither provide hand-labeled eye locations, nor are the images pre-cropped.
However, we want to use the same algorithms on those images as well, so we have to detect the face (and the facial landmarks), crop the face and perform a photometric enhancement.
So, image preprocessing becomes a three stage algorithm.

How to combine the two stages, image alignment and photometric enhancement, we have seen before.
Fortunately, the same technique can be applied for the :py:class:`bob.bio.face.preprocessor.FaceDetect`.
The face detector takes as an input a ``face_cropper``, where we can use the same options to select a face cropper, just that we cannot pass ``None``.
Interestingly, the face detector itself can be used as a ``face_cropper`` inside the photometric enhancement classes.
Hence, to generate a TanTriggs preprocessor that performs face detection, crops the face and performs photometric enhancement, you can create:

.. code-block:: py

   preprocessor = bob.bio.face.preprocessor.TanTriggs(face_cropper = bob.bio.face.preprocessor.FaceDetect(face_cropper = 'face-crop-eyes', use_flandmark = True) )

Or simply (using the face detector :ref:`Resource <bob.bio.face.preprocessors>`):

.. code-block:: py

   preprocessor = bob.bio.face.preprocessor.TanTriggs(face_cropper = 'landmark-detect')


.. _bob.bio.face.resources:

Registered Resources
--------------------

.. _bob.bio.face.databases:

Databases
~~~~~~~~~

One important aspect of :ref:`bob.bio.face <bob.bio.face>` is the relatively large list of supported image data sets, including well-defined evaluation protocols.
All databases rely on the :py:class:`bob.bio.base.database.DatabaseBob` interface, which in turn uses the :ref:`verification_databases`.
Please check the link above for information on how to obtain the original data of those data sets.

After downloading and extracting the original data of the data sets, it is necessary that the scripts know, where the data was installed.
For this purpose, the ``./bin/verify.py`` script can read a special file, where those directories are stored, see :ref:`bob.bio.base.installation`.
By default, this file is located in your home directory, but you can specify another file on command line.

The other option is to change the directories directly inside the configuration files.
Here is the list of files and replacement strings for all databases that are registered as resource, in alphabetical order:

* The AT&T database of faces: ``'atnt'``

  - Images: ``[YOUR_ATNT_DIRECTORY]``

* AR face: ``'arface'``

  - Images: ``[YOUR_ARFACE_DIRECTORY]``

* BANCA (english): ``'banca'``

  - Images: [YOUR_BANCA_DIRECTORY]

* CAS-PEAL: ``'caspeal'``

  - Images: ``[YOUR_CAS-PEAL_DIRECTORY]``

* Face Recognition Grand Challenge v2 (FRGC): ``'frgc'``

  - Complete directory: ``[YOUR_FRGC_DIRECTORY]``

  .. note::
     Due to implementation details, there will be a warning, when the FRGC database resource is loaded.
     To avoid this warning, you have to modify the FRGC database configuration file.

* The Good, the Bad and the Ugly (GBU): ``'gbu'``

  - Images (taken from MBGC-V1): ``[YOUR_MBGC-V1_DIRECTORY]``

* Labeled Faces in the Wild (LFW): ``'lfw-restricted'``, ``'lfw-unrestricted'``

  - Images (aligned with funneling): ``[YOUR_LFW_FUNNELED_DIRECTORY]``

  .. note::
     In the :ref:`bob.db.lfw <bob.db.lfw>` database interface, we provide automatically detected eye locations, which were detected on the funneled images.
     Face cropping using these eye locations will only work with the correct images.
     However, when using the face detector, all types of images will work.

* MOBIO: ``'mobio-image'``, ``'mobio-male'`` ``'mobio-female'``

  - Images (the .png images): ``[YOUR_MOBIO_IMAGE_DIRECTORY]``
  - Annotations (eyes): ``[YOUR_MOBIO_ANNOTATION_DIRECTORY]``

* Multi-PIE: ``'multipie'``, ``'multipie-pose'``

  - Images: ``[YOUR_MULTI-PIE_IMAGE_DIRECTORY]``
  - Annotations: ``[YOUR_MULTI-PIE_ANNOTATION_DIRECTORY]``

* SC face: ``'scface'``

  - Images: ``[YOUR_SC_FACE_DIRECTORY]``

* XM2VTS: ``'xm2vts'``

  - Images: ``[YOUR_XM2VTS_DIRECTORY]``


You can use the ``./bin/databases.py`` script to list, which data directories are correctly set up.

In order to view the annotations inside your database on top of the images, you can use the ``./bin/display_face_annotations.py`` script that is provided.
Please see ``./bin/display_face_annotations.py --help`` for more details and a list of options.


.. _bob.bio.face.preprocessors:

Preprocessors
~~~~~~~~~~~~~

Photometric enhancement algorithms are -- by default -- registered without face cropping, as ``'base'`` (no enhancement), ``'histogram'`` (histogram equalization), ``'tan-triggs'``, ``'self-quotient'`` (self quotient image) and ``'inorm-lbp'``.
These resources should only be used, when original images are already cropped (such as in the `AT&T database`_).

The default face cropping is performed by aligning the eye locations such that the eyes (in subject perspective) are located at: right eye: ``(16, 15)``, left eye: ``(16, 48)``, and the image is cropped to resolution ``(80, 64)`` pixels.
This cropper is registered under the resource key ``'face-crop-eyes'``.
Based on this cropping, photometric enhancement resources have a common addition: ``'histogram-crop'``, ``'tan-triggs-crop'``, ``'self-quotient-crop'`` and ``'inorm-lbp-crop'``.

For face detection, two resources are registered.
The ``'face-detect'`` resource will detect the face and perform ``'face-crop-eyes'``, without detecting the eye locations (fixed locations are taken instead).
Hence, the in-plane rotation of the face rotation not corrected by ``'face-detect'``.
On the other hand, in ``'landmark-detect'``, face detection and landmark localization are performed, and the face is aligned using ``'face-crop-eyes'``.
Photometric enhancement is only registered as resource after landmark localization: ``'histogram-landmark'``, ``'tan-triggs-landmark'``, ``'self-quotient-landmark'`` and ``'inorm-lbp-landmark'``.


.. _bob.bio.face.extractors:

Feature extractors
~~~~~~~~~~~~~~~~~~

Only four types of features are registered as resources here:

* ``'dct-blocks'``: DCT blocks with 12 pixels and full overlap, extracting 35 DCT features per block
* ``'eigenface'``: Pixel vectors projected to face space keeping 95 % variance
* ``'grid-graph'``: Gabor jets in grid graphs, with 8 pixels distance between nodes
* ``'lgbphs'``: Local Gabor binary pattern histogram sequences with block-size of 8 and no overlap

.. _bob.bio.face.algorithms:

Face Recognition Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``'gabor-jet'``: Compares graphs of Gabor jets with using a dedicated Gabor jet similarity function [GHW12]_
* ``'histogram'``: Compares histograms using histogram comparison functions
* ``'bic-jet'``: Uses the :py:class:`bob.bio.base.algorithm.BIC` with vectors of Gabor jet similarities

  .. note:: One particularity of this resource is that the function to compute the feature vectors to be classified in the BIC algorithm is actually implemented *in the configuration file*.


.. include:: links.rst
