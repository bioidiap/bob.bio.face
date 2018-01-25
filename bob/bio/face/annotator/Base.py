from bob.bio.base import read_original_data as base_read


class Base(object):
    """Base class for all annotators"""

    def __init__(self, read_original_data=None, **kwargs):
        super(Base, self).__init__(**kwargs)
        self.read_original_data = read_original_data or base_read

    def annotate(self, image, **kwargs):
        """Annotates an image and returns annotations in a dictionary.
        All annotator should return at least the topleft and bottomright
        coordinates.

        Parameters
        ----------
        image : array
            The image should be a Bob format (#Channels, Height, Width) RGB
            image.
        **kwargs
            The extra arguments that may be passed.
        """
        raise NotImplementedError()

    # Alisa call to annotate
    def __call__(self, image, **kwargs):
        return self.annotate(image, **kwargs)
