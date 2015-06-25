from .Base import Base
from .FaceCrop import FaceCrop
from .FaceDetect import FaceDetect

from .TanTriggs import TanTriggs
from .INormLBP import INormLBP
from .HistogramEqualization import HistogramEqualization
from .SelfQuotientImage import SelfQuotientImage

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
