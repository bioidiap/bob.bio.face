.. vim: set fileencoding=utf-8 :

.. _bob.bio.face.annotators:

=================
 Face Annotators
=================

This packages provides several face annotators (using RGB images) that you can
use to annotate biometric databases. See :ref:`bob.bio.base.annotators` for
a guide on the general usage of this feature.

.. warning::

    The annotators are named after the package that provides this annotator.
    Their respective |project| packages need to be installed if you want to use
    them.

See :doc:`implemented` for a list of available annotators. Here is an example
on how to use :any:`bob.bio.face.annotator.MTCNN` to annotate the ATNT
database:

.. code-block:: sh

    $ bob bio annotate -vvv -d atnt -a mtcnn -o /tmp/annotations
