# isort: skip_file
from . import preprocessor
from . import algorithm
from . import script
from . import database
from . import annotator
from . import utils
from . import pytorch
from . import embeddings

from . import test


def get_config():
    """Returns a string containing the configuration information."""

    import bob.extension

    return bob.extension.get_config(__name__)


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith("_")]
