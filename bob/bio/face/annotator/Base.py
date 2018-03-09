from bob.bio.base.annotator import Annotator
from bob.bio.face.preprocessor import FaceCrop  # import for documentation


class Base(Annotator):
    """Base class for all face annotators"""

    def __init__(self, **kwargs):
        super(Base, self).__init__(**kwargs)

    def annotate(self, sample, **kwargs):
        """Annotates an image and returns annotations in a dictionary. All
        annotator should return at least the ``topleft`` and ``bottomright``
        coordinates. Some currently known annotation points such as ``reye``
        and ``leye`` are formalized in
        :any:`FaceCrop`.

        Parameters
        ----------
        sample : numpy.ndarray
            The image should be a Bob format (#Channels, Height, Width) RGB
            image.
        **kwargs
            The extra arguments that may be passed.
        """
        raise NotImplementedError()

    # Alisa call to annotate
    def __call__(self, sample, **kwargs):
        return self.annotate(sample, **kwargs)
