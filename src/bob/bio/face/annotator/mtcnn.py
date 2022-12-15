# Example taken from:
# https://github.com/blaueck/tf-mtcnn/blob/master/mtcnn_tfv2.py

import logging

import pkg_resources

from bob.bio.face.color import gray_to_rgb
from bob.io.image import to_matplotlib

from . import Base

logger = logging.getLogger(__name__)


class MTCNN(Base):

    """MTCNN v1 wrapper for Tensorflow 2. See
    https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html for
    more details on MTCNN.

    Attributes
    ----------
    factor : float
        Factor is a trade-off between performance and speed.
    min_size : int
        Minimum face size to be detected.
    thresholds : list
        Thresholds are a trade-off between false positives and missed detections.
    """

    def __init__(
        self, min_size=40, factor=0.709, thresholds=(0.6, 0.7, 0.7), **kwargs
    ):
        super().__init__(**kwargs)
        self.min_size = min_size
        self.factor = factor
        self.thresholds = thresholds
        self._graph_path = pkg_resources.resource_filename(
            "bob.bio.face", "data/mtcnn.pb"
        )

        # Avoids loading graph at initilization
        self._fun = None

    @property
    def mtcnn_fun(self):
        import tensorflow as tf

        if self._fun is None:
            # wrap graph function as a callable function
            self._fun = tf.compat.v1.wrap_function(
                self._graph_fn,
                [
                    tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                ],
            )
        return self._fun

    def _graph_fn(self, img):
        import tensorflow as tf

        with open(self._graph_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef.FromString(f.read())

        prob, landmarks, box = tf.compat.v1.import_graph_def(
            graph_def,
            input_map={
                "input:0": img,
                "min_size:0": tf.convert_to_tensor(self.min_size, dtype=float),
                "thresholds:0": tf.convert_to_tensor(
                    self.thresholds, dtype=float
                ),
                "factor:0": tf.convert_to_tensor(self.factor, dtype=float),
            },
            return_elements=["prob:0", "landmarks:0", "box:0"],
            name="",
        )
        return box, prob, landmarks

    def __getstate__(self):
        # Handling unpicklable objects
        state = {}
        for key, value in super().__getstate__().items():
            if key != "_fun":
                state[key] = value
        state["_fun"] = None
        return state

    def detect(self, image):
        """Detects all faces in the image.

        Parameters
        ----------
        image : numpy.ndarray
            An RGB image in Bob format.

        Returns
        -------
        tuple
            A tuple of boxes, probabilities, and landmarks.
        """
        if len(image.shape) == 2:
            image = gray_to_rgb(image)

        # Assuming image is Bob format and RGB
        assert image.shape[0] == 3, image.shape
        # MTCNN expects BGR opencv format
        image = to_matplotlib(image)
        image = image[..., ::-1]

        boxes, probs, landmarks = self.mtcnn_fun(image)
        return boxes, probs, landmarks

    def annotations(self, image):
        """Detects all faces in the image and returns annotations in bob format.

        Parameters
        ----------
        image : numpy.ndarray
            An RGB image in Bob format.

        Returns
        -------
        list
            A list of annotations. Annotations are dictionaries that contain the
            following keys: ``topleft``, ``bottomright``, ``reye``, ``leye``, ``nose``,
            ``mouthright``, ``mouthleft``, and ``quality``.
        """
        boxes, probs, landmarks = self.detect(image)

        # Iterate over all the detected faces
        annots = []
        for box, prob, lm in zip(boxes, probs, landmarks):
            topleft = float(box[0]), float(box[1])
            bottomright = float(box[2]), float(box[3])
            right_eye = float(lm[0]), float(lm[5])
            left_eye = float(lm[1]), float(lm[6])
            nose = float(lm[2]), float(lm[7])
            mouthright = float(lm[3]), float(lm[8])
            mouthleft = float(lm[4]), float(lm[9])
            annots.append(
                {
                    "topleft": topleft,
                    "bottomright": bottomright,
                    "reye": right_eye,
                    "leye": left_eye,
                    "nose": nose,
                    "mouthright": mouthright,
                    "mouthleft": mouthleft,
                    "quality": float(prob),
                }
            )
        return annots

    def annotate(self, image, **kwargs):
        """Annotates an image using mtcnn

        Parameters
        ----------
        image : numpy.array
            An RGB image in Bob format.
        **kwargs
            Ignored.

        Returns
        -------
        dict
            Annotations contain: (topleft, bottomright, leye, reye, nose,
            mouthleft, mouthright, quality).
        """
        # return the annotations for the first/largest face
        annotations = self.annotations(image)

        if annotations:
            return annotations[0]
        else:
            return None
