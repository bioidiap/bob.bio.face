.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <siebenkopf@googlemail.com>

.. _bob.bio.face.datasets:

Setting up Facial Image Databases
=================================

This package includes a large list of facial image datasets including their default evaluation protocols.
You can list all available datasets and protocols by calling:

.. code-block:: bash

  resources.py --types databases


We use the default bob configuration system for setting up database interfaces, which basically translates to:

.. code-block:: bash

  bob config set <key> <value>

Mainly, we need to tell the system where the original files can be located.
In some circumstances, the filename extension of the datasets can change from version to version, so they can be setup as well.
In the following, find a list of all current databases, their configuration parameters, and their provided protocols, in alphabetic order.

* `ARface dataset <https://www2.ece.ohio-state.edu/~aleix/ARdatabase.html>`__

  - directory key: ``bob.bio.face.database.arface.directory``; the directory containing all the images
  - extension key: ``bob.bio.face.database.arface.extension``; ``".ppm"`` if not set
  - protocols: ``expression, illumination, occlusion, occlusion_and_illumination, all``
  - annotations: eye centers (provided in the interface)
  - notes: The original files are stored in a raw format and need to be converted into some image format.

* AT&T dataset of faces, previously known as the ORL dataset

  - annotations: none, faces are already aligned
  - notes: The dataset will be downloaded completely, including all images. No special setup required.
