import logging
import os
import sys

from itertools import product as product
from math import ceil

import numpy as np
import torch

from torchvision import transforms

from bob.extension.download import get_file
from bob.io.image import bob_to_opencvbgr

from . import Base
from .mtcnn import MTCNN

logger = logging.getLogger(__name__)


# Adapted from https://github.com/biubug6/Pytorch_Retinafacey
class PriorBox(object):
    """Compute the suitable parameters of anchors for later decode operation

    Attributes:
        cfg(dict): testing config.
        image_size(tuple): the input image size.
    """

    def __init__(self, cfg, image_size=None):
        """
        Init priorBox settings related to the generation of anchors.
        """
        super(PriorBox, self).__init__()
        self.min_sizes = cfg["min_sizes"]
        self.steps = cfg["steps"]
        self.image_size = image_size
        self.feature_maps = [
            [ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)]
            for step in self.steps
        ]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [
                        x * self.steps[k] / self.image_size[1]
                        for x in [j + 0.5]
                    ]
                    dense_cy = [
                        y * self.steps[k] / self.image_size[0]
                        for y in [i + 0.5]
                    ]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        return output


def download_faceX_model():
    urls = [
        "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/faceX_models.tar.gz",
        "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/faceX_models.tar.gz",
    ]

    filename = get_file(
        "faceX_models.tar.gz",
        urls,
        cache_subdir="data/pytorch/",
        file_hash="eb7ec871f434d2f44e5408627d656297",
        extract=True,
    )

    return filename


def add_faceX_path(filename):

    path = os.path.join(os.path.dirname(filename), "faceX_models")

    logger.warning(f"Adding the following path to PYTHON_PATH: {path}")
    sys.path.insert(0, path)
    return path


class FaceXDetector(Base):
    """
    Face detector taken from https://github.com/JDAI-CV/FaceX-Zoo

    This one we are using the 106 larnmark detector that was taken from
    https://github.com/Hsintao/pfld_106_face_landmarks/blob/master/models/mobilev3_pfld.py

    .. warning:
      Here we are assuming that the faces is already detected and cropped

    """

    def __init__(self, device=None, one_face_only=True, **kwargs):
        self.device = torch.device("cpu") if device is None else device

        filename = download_faceX_model()
        faceX_path = add_faceX_path(filename)

        model_filename = os.path.join(
            faceX_path,
            "models",
            "face_detection",
            "face_detection_1.0",
            "face_detection_retina.pkl",
        )

        # Loading face detector
        self.model = torch.load(model_filename, map_location=device)
        self.one_face_only = one_face_only

        self.transforms = transforms.Compose([transforms.ToTensor()])

        # Face detection threshold
        # from: https://github.com/JDAI-CV/FaceX-Zoo/blob/db0b087e4f4d28152e172d6c8d3767a8870733b4/face_sdk/models/face_detection/face_detection_1.0/model_meta.json
        self.cfg = {
            "model_type": "retina face detect nets",
            "model_info": "some model info",
            "model_file": "face_detection_retina.pkl",
            "release_date": "20201019",
            "input_height": 120,
            "input_width": 120,
            "min_sizes": [[16, 32], [64, 128], [256, 512]],
            "steps": [8, 16, 32],
            "variance": [0.1, 0.2],
            "in_channel": 256,
            "out_channel": 256,
            "confidence_threshold": 0.7,
        }

        super(FaceXDetector, self).__init__(**kwargs)

    # Adapted from https://github.com/chainer/chainercv
    def decode(self, loc, priors, variances):
        """

        Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.

        Parameters
        ----------
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes

        Returns
        -------
            decoded bounding box predictions

        """
        boxes = torch.cat((priors[:, :2], priors[:, 2:]), 1)
        boxes[:, :2] = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
        boxes[:, 2:] = priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def _preprocess(self, image):
        """Preprocess the image, such as standardization and other operations.

        Returns:
            A numpy array list, the shape is channel * h * w.
            A tensor, the shape is 4.
        """
        if not isinstance(image, np.ndarray):
            logger.error("The input should be the ndarray read by cv2!")

        img = np.float32(image)
        scale = torch.Tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
        )
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        return img, scale

    def _postprocess(self, loc, conf, scale, input_height, input_width):
        """Postprecess the prediction result.
        Decode detection result, set the confidence threshold and do the NMS
        to keep the appropriate detection box.

        Returns:
            A numpy array, the shape is N * (x, y, w, h, confidence),
            N is the number of detection box.
        """
        priorbox = PriorBox(self.cfg, image_size=(input_height, input_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = self.decode(
            loc.data.squeeze(0), prior_data, self.cfg["variance"]
        )
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > self.cfg["confidence_threshold"])[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        nms_threshold = 0.2
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False
        )
        keep = self.py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        return dets

    # Adapted from https://github.com/biubug6/Pytorch_Retinaface
    def py_cpu_nms(self, dets, thresh):
        """Python version NMS (Non maximum suppression).

        Returns:
            The kept index after NMS.
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    def annotate(self, image, **kwargs):
        """Get the inference of the image and process the inference result.

        Returns:
            A numpy array, the shape is N * (x, y, w, h, confidence),
            N is the number of detection box.
        """

        # First thing, we need to convert the bob CxHxW
        # to the openCV HxWxC and BGR
        image = bob_to_opencvbgr(image)

        input_height, input_width, _ = image.shape
        try:
            image, scale = self._preprocess(image)
        except Exception as e:
            raise e
        self.model = self.model.to(self.device)
        image = torch.from_numpy(image).unsqueeze(0)
        with torch.no_grad():
            image = image.to(self.device)
            scale = scale.to(self.device)
            loc, conf, landms = self.model(image)
        dets = self._postprocess(loc, conf, scale, input_height, input_width)

        if len(dets) == 0:
            logger.error("Face not detected. Returning None")
            return None

        dets = dets[0] if self.one_face_only else dets

        return dets


class FaceX106Landmarks(Base):
    """
    Landmark detector taken from https://github.com/JDAI-CV/FaceX-Zoo

    This one we are using the 106 larnmark detector that was taken from
    https://github.com/Hsintao/pfld_106_face_landmarks/blob/master/models/mobilev3_pfld.py

    .. warning:
      Here we are assuming that the faces is already detected and cropped


    Parameters
    ----------

    use_mtcnn_detector: bool
       If set uses the MTCNN face detector as a base for the landmark extractor.
       If not, it uses the standard face detector of FaceXZoo.


    """

    def __init__(self, device=None, use_mtcnn_detector=True, **kwargs):
        self.device = torch.device("cpu") if device is None else device

        filename = download_faceX_model()
        faceX_path = add_faceX_path(filename)
        self.use_mtcnn_detector = use_mtcnn_detector

        model_filename = os.path.join(
            faceX_path,
            "models",
            "face_alignment",
            "face_alignment_1.0",
            "face_landmark_pfld.pkl",
        )

        self.model = torch.load(model_filename, map_location=self.device)

        # Loading the face detector
        self.face_detector = MTCNN() if use_mtcnn_detector else FaceXDetector()

        self.transforms = transforms.Compose([transforms.ToTensor()])

        # Face alignment threshold
        # from: https://github.com/JDAI-CV/FaceX-Zoo/blob/db0b087e4f4d28152e172d6c8d3767a8870733b4/face_sdk/models/face_alignment/face_alignment_1.0/model_meta.json
        self.cfg = {
            "model_path": "models",
            "model_category": "face_alignment",
            "model_name": "face_alignment_1.0",
            "model_type": "pfld face landmark nets",
            "model_info": "some model info",
            "model_file_path": "models/face_alignment/face_alignment_1.0/face_landmark_pfld.pkl",
            "release_date": "20201023",
            "input_height": 112,
            "input_width": 112,
            "img_size": 112,
        }

        self.img_size = self.cfg["img_size"]

        super(FaceX106Landmarks, self).__init__(**kwargs)

        # self.detector = MTCNN(min_size=min_size, factor=factor, thresholds=thresholds)

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

        # Detect the face
        if self.use_mtcnn_detector:
            annotations = self.face_detector.annotate(image)
            if annotations is None:
                return None

            dets = [
                annotations["topleft"][1],
                annotations["topleft"][0],
                annotations["bottomright"][1],
                annotations["bottomright"][0],
            ]
        else:
            dets = self.face_detector.annotate(image.copy())

        if dets is None:
            return None

        # First thing, we need to convert the bob CxHxW
        # to the openCV HxWxC and BGR
        image = bob_to_opencvbgr(image)
        try:
            image_pre = self._preprocess(image, dets)
        except Exception as e:
            raise e
        self.model = self.model.to(self.device)
        image_pre = image_pre.unsqueeze(0)
        with torch.no_grad():
            image_pre = image_pre.to(self.device)
            _, landmarks_normal = self.model(image_pre)
        landmarks = self._postprocess(landmarks_normal)

        return np.array(landmarks)

    # Adapted from https://github.com/Hsintao/pfld_106_face_landmarks/blob/master/data/prepare.py
    def _preprocess(self, image, det):
        import cv2

        """Preprocess the input image, cutting the input image through the face detection information.
        Using the face detection result(dets) to get the face position in the input image.
        After determining the center of face position and the box size of face, crop the image
        and resize it into preset size.

        Returns:
           A torch tensor, the image after preprecess, shape: (3, 112, 112).
        """
        if not isinstance(image, np.ndarray):
            logger.error("The input should be the ndarray read by cv2!")

        img = image.copy()
        self.image_org = image.copy()
        img = np.float32(img)

        xy = np.array([det[0], det[1]])
        zz = np.array([det[2], det[3]])
        wh = zz - xy + 1
        center = (xy + wh / 2).astype(np.int32)
        boxsize = int(np.max(wh) * 1.2)
        xy = center - boxsize // 2
        self.xy = xy
        self.boxsize = boxsize
        x1, y1 = xy
        x2, y2 = xy + boxsize
        height, width, _ = img.shape
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)
        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)
        imageT = image[y1:y2, x1:x2]
        if dx > 0 or dy > 0 or edx > 0 or edy > 0:
            imageT = cv2.copyMakeBorder(
                imageT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0
            )

        imageT = cv2.resize(imageT, (self.img_size, self.img_size))
        t = transforms.Compose([transforms.ToTensor()])
        img_after = t(imageT)
        return img_after

    def _postprocess(self, landmarks_normal):
        """Process the predicted landmarks into the form of the original image.

        Returns:
            A numpy array, the landmarks based on the shape of original image, shape: (106, 2),
        """
        landmarks_normal = landmarks_normal.cpu().numpy()
        landmarks_normal = landmarks_normal.reshape(
            landmarks_normal.shape[0], -1, 2
        )
        landmarks = landmarks_normal[0] * [self.boxsize, self.boxsize] + self.xy
        return landmarks
