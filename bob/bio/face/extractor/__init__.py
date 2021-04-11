from .DCTBlocks import DCTBlocks
from .GridGraph import GridGraph
from .LGBPHS import LGBPHS
from .mxnet_resnet import mxnet_model
from .pytorch_model import pytorch_loaded_model
from .pytorch_model import pytorch_library_model
from .tf_model import tf_model
from .opencv_caffe import opencv_model

# gets sphinx autodoc done right - don't remove it
def __appropriate__(*args):
  """Says object was actually declared here, and not in the import module.
  Fixing sphinx warnings of not being able to find classes, when path is shortened.
  Parameters:

    *args: An iterable of objects to modify

  Resolves `Sphinx referencing issues
  <https://github.com/sphinx-doc/sphinx/issues/3048>`
  """

  for obj in args: obj.__module__ = __name__

__appropriate__(
    DCTBlocks,
    GridGraph,
    LGBPHS,
    mxnet_model,
    pytorch_loaded_model,
    pytorch_library_model,
    tf_model,
    opencv_model,
    )
__all__ = [_ for _ in dir() if not _.startswith('_')]
