import os
import bob.extension.download

def download_model(model_path, urls, zip_file="model.tar.gz"):
    """
    Download and unzip a model from some URL.

    Parameters
    ----------

    model_path: str
        Path where the model is supposed to be stored

    urls: list
        List of paths where the model is stored

    zip_file: str
        File name after the download

    """

    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
        zip_file = os.path.join(model_path, zip_file)
        bob.extension.download.download_and_unzip(urls, zip_file)


from .tf2_inception_resnet import (
    InceptionResnet,
    InceptionResnetv2_MsCeleb_CenterLoss_2018,
    InceptionResnetv2_Casia_CenterLoss_2018,
    InceptionResnetv1_MsCeleb_CenterLoss_2018,
    InceptionResnetv1_Casia_CenterLoss_2018,
    FaceNetSanderberg_20170512_110547
)

# gets sphinx autodoc done right - don't remove it
def __appropriate__(*args):
    """Says object was actually declared here, and not in the import module.
  Fixing sphinx warnings of not being able to find classes, when path is shortened.
  Parameters:

    *args: An iterable of objects to modify

  Resolves `Sphinx referencing issues
  <https://github.com/sphinx-doc/sphinx/issues/3048>`
  """

    for obj in args:
        obj.__module__ = __name__


__appropriate__(
    InceptionResnet,
    InceptionResnetv2_MsCeleb_CenterLoss_2018,
    InceptionResnetv1_MsCeleb_CenterLoss_2018,
    InceptionResnetv2_Casia_CenterLoss_2018,
    InceptionResnetv1_Casia_CenterLoss_2018,
    FaceNetSanderberg_20170512_110547
)
__all__ = [_ for _ in dir() if not _.startswith("_")]
