import bob.bio.base.annotator

from bob.bio.base.annotator.FailSafe import translate_kwargs


class Base(bob.bio.base.annotator.Annotator):
    """Base class for all face annotators"""

    def annotations(self, image):
        """Returns annotations for all faces in the image.

        Parameters
        ----------
        image : numpy.ndarray
            An RGB image in Bob format.

        Returns
        -------
        list
            A list of annotations. Annotations are dictionaries that contain the
            following possible keys: ``topleft``, ``bottomright``, ``reye``, ``leye``
        """
        raise NotImplementedError()

    def annotate(self, sample, **kwargs):
        """Annotates an image and returns annotations in a dictionary. All
        annotator should return at least the ``topleft`` and ``bottomright``
        coordinates. Some currently known annotation points such as ``reye``
        and ``leye`` are formalized in
        :any:`bob.bio.face.preprocessor.FaceCrop`.

        Parameters
        ----------
        sample : numpy.ndarray
            The image should be a Bob format (#Channels, Height, Width) RGB
            image.
        **kwargs
            The extra arguments that may be passed.
        """
        raise NotImplementedError()

    def transform(self, samples, **kwargs):
        """Annotates an image and returns annotations in a dictionary.

        All annotator should add at least the ``topleft`` and ``bottomright``
        coordinates. Some currently known annotation points such as ``reye``
        and ``leye`` are formalized in
        :any:`bob.bio.face.preprocessor.FaceCrop`.

        Parameters
        ----------
        sample : Sample
            The image int the sample object should be a Bob format
            (#Channels, Height, Width) RGB image.
        **kwargs
            Extra arguments that may be passed.
        """
        kwargs = translate_kwargs(kwargs, len(samples))
        return [
            self.annotate(sample, **kw) for sample, kw in zip(samples, kwargs)
        ]
