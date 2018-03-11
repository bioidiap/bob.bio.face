from . import Base
from bob.ip.color import rgb_to_gray
from bob.ip.flandmark import Flandmark


class BobIpFlandmark(Base):
    """Annotator using bob.ip.flandmark.
    This annotator needs the topleft and bottomright annotations provided."""

    def __init__(self, **kwargs):
        super(BobIpFlandmark, self).__init__(**kwargs)
        self.flandmark = Flandmark()

    def annotate(self, image, annotations, **kwargs):
        """Annotates an image.

        Parameters
        ----------
        image : array
            Image in Bob format RGB.
        annotations : dict
            The topleft and bottomright annotations are required.
        **kwargs
            Ignored.

        Returns
        -------
        dict
            Annotations with reye and leye keys or None if it fails.
        """
        image = rgb_to_gray(image)
        top, left = annotations['topleft']
        top, left = int(max(top, 0)), int(max(left, 0))
        height = annotations['bottomright'][0] - top
        width = annotations['bottomright'][1] - left
        height, width = min(height, image.shape[0]), min(width, image.shape[1])
        height, width = int(height), int(width)

        landmarks = self.flandmark.locate(image, top, left, height, width)

        if landmarks is not None and len(landmarks):
            return {
                'reye': ((landmarks[1][0] + landmarks[5][0]) / 2.,
                         (landmarks[1][1] + landmarks[5][1]) / 2.),
                'leye': ((landmarks[2][0] + landmarks[6][0]) / 2.,
                         (landmarks[2][1] + landmarks[6][1]) / 2.)
            }
        else:
            return None
