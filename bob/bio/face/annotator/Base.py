from bob.bio.base import read_original_data as base_read


class Base(object):
    """Base class for all annotators"""

    def __init__(self, read_original_data=None, **kwargs):
        super(Base, self).__init__(**kwargs)
        self.read_original_data = read_original_data or base_read

    def annotate(self, image, **kwargs):
        """Annotates an image and returns annotations in a dictionary

        Parameters
        ----------
        image : object
            The image is what comes out of ``read_original_data``.
        **kwargs
            The extra arguments that may be passed.
        """
        raise NotImplementedError()

    # Alisa call to annotate
    __call__ = annotate
    __call__.__doc__ = annotate.__doc__
