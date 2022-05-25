# isort: skip_file
from . import preprocessor  # noqa: F401
from . import script  # noqa: F401
from . import database  # noqa: F401
from . import annotator  # noqa: F401
from . import utils  # noqa: F401
from . import pytorch  # noqa: F401
from . import embeddings  # noqa: F401
from . import test  # noqa: F401


def get_config():
    """Returns a string containing the configuration information."""

    import bob.extension

    return bob.extension.get_config(__name__)


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith("_")]
