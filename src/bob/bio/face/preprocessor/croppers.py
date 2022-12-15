#!/usr/bin/env python
# @author: Tiago de Freitas Pereira

"""
Implements some face croppers
"""


import logging

from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger("bob.bio.face")

import cv2
import numpy as np

from bob.io.image import bob_to_opencvbgr, opencvbgr_to_bob


class FaceEyesNorm(TransformerMixin, BaseEstimator):
    """
    Geometric normalize a face using the eyes positions
    This function extracts the facial image based on the eye locations (or the location of other fixed point, see note below). "
    The geometric normalization is applied such that the eyes are placed to **fixed positions** in the normalized image.
    The image is cropped at the same time, so that no unnecessary operations are executed.

    There are three types of annotations:
        - **eyes-center**: The eyes are located at the center of the face. In this case, `reference_eyes_location` expects
            a dictionary with two keys: `leye` and `reye`.

        - **left-profile**: The eyes are located at the corner of the face. In this case, `reference_eyes_location` expects
            a dictionary with two keys: `leye` and `mouth`.

        - **right-profile**: The eyes are located at the corner of the face. In this case, `reference_eyes_location` expects
            a dictionary with two keys: `reye` and `mouth`.


    Parameters
    ----------

        reference_eyes_location : dict
            The reference eyes location. It is a dictionary with two keys.

        final_image_size : tuple
            The final size of the image

        allow_upside_down_normalized_faces: bool
            If set to True, the normalized face will be flipped if the eyes are placed upside down.

        annotation_type : str
            The type of annotation. It can be either 'eyes-center' or 'left-profile' or 'right-profile'

        opencv_interpolation : int
            The interpolation method to be used by OpenCV for the function cv2.warpAffine


    """

    def __init__(
        self,
        reference_eyes_location,
        final_image_size,
        allow_upside_down_normalized_faces=False,
        annotation_type="eyes-center",
        opencv_interpolation=cv2.INTER_LINEAR,
    ):
        self.annotation_type = annotation_type
        self.reference_eyes_location = reference_eyes_location
        self._check_annotations(self.reference_eyes_location)
        self.opencv_interpolation = opencv_interpolation

        self.allow_upside_down_normalized_faces = (
            allow_upside_down_normalized_faces
        )
        (
            self.target_eyes_distance,
            self.target_eyes_center,
            self.target_eyes_angle,
        ) = self._get_anthropometric_measurements(reference_eyes_location)

        self.final_image_size = final_image_size

    def _check_annotations(self, positions):

        if self.annotation_type == "eyes-center":
            assert "leye" in positions
            assert "reye" in positions
        elif self.annotation_type == "left-profile":
            assert "leye" in positions
            assert "mouth" in positions
        elif self.annotation_type == "right-profile":
            assert "reye" in positions
            assert "mouth" in positions
        else:
            raise ValueError(
                "The annotation type must be either 'eyes-center', 'left-profile' or 'right-profile'"
            )

    def _decode_positions(self, positions):
        """
        Return the annotation positions, based on the annotation type
        """
        if self.annotation_type == "eyes-center":
            return np.array(positions["leye"]), np.array(positions["reye"])
        elif self.annotation_type == "left-profile":
            return np.array(positions["leye"]), np.array(positions["mouth"])
        elif self.annotation_type == "right-profile":
            return np.array(positions["reye"]), np.array(positions["mouth"])
        else:
            raise ValueError(
                "The annotation type must be either 'eyes-center', 'left-profile' or 'right-profile'"
            )

    def _get_anthropometric_measurements(self, positions):
        """
        Given the eyes coordinates, it computes the
         - The angle between the eyes coordinates
         - The distance between the eyes coordinates
         - The center of the eyes coordinates

        """

        # double dy = leftEye[0] - rightEye[0], dx = leftEye[1] - rightEye[1];
        # double angle = std::atan2(dy, dx);
        coordinate_a, coordinate_b = self._decode_positions(positions)
        delta = coordinate_a - coordinate_b
        eyes_angle = np.arctan2(delta[0], delta[1]) * 180 / np.pi  # to degrees

        # Or scaling factor
        # m_eyesDistance / sqrt(_sqr(leftEye[0]-rightEye[0]) + _sqr(leftEye[1]-rightEye[1]))
        eyes_distance = np.linalg.norm(delta)

        eyes_center = 1 / 2 * (coordinate_a + coordinate_b)

        return eyes_distance, eyes_center, eyes_angle

    def _more_tags(self):
        return {"requires_fit": False}

    def fit(self, X, y=None):
        return self

    def _check_upsidedown(self, annotations):
        coordinate_a, coordinate_b = self._decode_positions(annotations)
        reference_coordinate_a, reference_coordinate_b = self._decode_positions(
            self.reference_eyes_location
        )

        reye_desired_width = reference_coordinate_a[1]
        leye_desired_width = reference_coordinate_b[1]
        right_eye = coordinate_a
        left_eye = coordinate_b

        if (
            reye_desired_width > leye_desired_width
            and right_eye[1] < left_eye[1]
        ) or (
            reye_desired_width < leye_desired_width
            and right_eye[1] > left_eye[1]
        ):
            raise ValueError(
                "Looks like 'leye' and 'reye' in annotations: {annot} are swapped. "
                "This will make the normalized face upside down (compared to the original "
                "image). Most probably your annotations are wrong. Otherwise, you can set "
                "the ``allow_upside_down_normalized_faces`` parameter to "
                "True.".format(annot=annotations)
            )

    def _rotate_image_center(self, image, angle, reference_point):
        """
        Rotate the image around the center by the given angle.
        """

        rot_mat = cv2.getRotationMatrix2D(reference_point[::-1], angle, 1.0)

        return cv2.warpAffine(
            image, rot_mat, image.shape[1::-1], flags=self.opencv_interpolation
        )

    def _translate_image(self, image, x, y):
        t_mat = np.float32([[1, 0, x], [0, 1, y]])
        return cv2.warpAffine(
            image, t_mat, image.shape[1::-1], flags=self.opencv_interpolation
        )

    def transform(self, X, annotations=None):
        """
        Geometric normalize a face using the eyes positions

        Parameters
        ----------

        X : numpy.ndarray
            The image to be normalized

        annotations : dict
            The annotations of the image. It needs to contain ''reye'' and ''leye'' positions


        Returns
        -------

            cropped_image : numpy.ndarray
                The normalized image

        """

        self._check_annotations(annotations)

        if not self.allow_upside_down_normalized_faces:
            self._check_upsidedown(annotations)

        (
            source_eyes_distance,
            source_eyes_center,
            source_eyes_angle,
        ) = self._get_anthropometric_measurements(annotations)

        # m_geomNorm->setRotationAngle(angle * 180. / M_PI - m_eyesAngle);
        # Computing the rotation angle with respect to the target eyes angle in degrees
        rotational_angle = source_eyes_angle - self.target_eyes_angle

        # source_target_ratio = source_eyes_distance / self.target_eyes_distance
        target_source_ratio = self.target_eyes_distance / source_eyes_distance

        #

        # ROTATION WITH OPEN CV

        cropped_image = bob_to_opencvbgr(X) if X.ndim > 2 else X
        original_height = cropped_image.shape[0]
        original_width = cropped_image.shape[1]

        cropped_image = self._rotate_image_center(
            cropped_image, rotational_angle, source_eyes_center
        )

        # Cropping

        target_eyes_center_rescaled = np.floor(
            self.target_eyes_center / target_source_ratio
        ).astype("int")

        top = int(source_eyes_center[0] - target_eyes_center_rescaled[0])
        left = int(source_eyes_center[1] - target_eyes_center_rescaled[1])

        bottom = max(0, top) + (
            int(self.final_image_size[0] / target_source_ratio)
        )
        right = max(0, left) + (
            int(self.final_image_size[1] / target_source_ratio)
        )

        cropped_image = cropped_image[
            max(0, top) : bottom, max(0, left) : right, ...
        ]

        # Checking if we need to pad the cropped image
        # This happens when the cropped image extrapolate the original image dimensions
        expanded_image = cropped_image

        if original_height < bottom or original_width < right:

            pad_height = (
                cropped_image.shape[0] + (bottom - original_height)
                if original_height < bottom
                else cropped_image.shape[0]
            )

            pad_width = (
                cropped_image.shape[1] + (right - original_width)
                if original_width < right
                else cropped_image.shape[1]
            )

            expanded_image = (
                np.zeros(
                    (pad_height, pad_width, 3),
                    dtype=cropped_image.dtype,
                )
                if cropped_image.ndim > 2
                else np.zeros(
                    (pad_height, pad_width), dtype=cropped_image.dtype
                )
            )

            expanded_image[
                0 : cropped_image.shape[0], 0 : cropped_image.shape[1], ...
            ] = cropped_image

        # Checking if we need to translate the image.
        # This happens when the top, left coordinates on the source images is negative
        if top < 0 or left < 0:
            expanded_image = self._translate_image(
                expanded_image, -1 * min(0, left), -1 * min(0, top)
            )

        # Scaling

        expanded_image = cv2.resize(
            expanded_image,
            self.final_image_size[::-1],
            interpolation=self.opencv_interpolation,
        )

        expanded_image = (
            opencvbgr_to_bob(expanded_image) if X.ndim > 2 else expanded_image
        )

        return expanded_image


class FaceCropBoundingBox(TransformerMixin, BaseEstimator):
    """
    Crop the face based on Bounding box positions

    Parameters
    ----------

    final_image_size : tuple
        The final size of the image after cropping in case resize=True

    margin : float
        The margin to be added to the bounding box


    """

    def __init__(
        self,
        final_image_size,
        margin=0.5,
        opencv_interpolation=cv2.INTER_LINEAR,
    ):
        self.margin = margin
        self.final_image_size = final_image_size
        self.opencv_interpolation = opencv_interpolation

    def transform(self, X, annotations, resize=True):
        """
        Crop the face based on Bounding box positions

        Parameters
        ----------

        X : numpy.ndarray
            The image to be normalized

        annotations : dict
            The annotations of the image. It needs to contain ''topleft'' and ''bottomright'' positions

        resize: bool
            If True, the image will be resized to the final size
            In this case, margin is not used

        """

        assert "topleft" in annotations
        assert "bottomright" in annotations

        # If it's grayscaled, expand dims
        if X.ndim == 2:
            logger.warning(
                "Gray-scaled image. Expanding the channels before detection"
            )
            X = np.repeat(np.expand_dims(X, 0), 3, axis=0)

        top = int(annotations["topleft"][0])
        left = int(annotations["topleft"][1])

        bottom = int(annotations["bottomright"][0])
        right = int(annotations["bottomright"][1])

        width = right - left
        height = bottom - top

        if resize:
            # If resizing, don't use the expanded borders
            face_crop = X[
                :,
                top:bottom,
                left:right,
            ]

            face_crop = (
                bob_to_opencvbgr(face_crop) if face_crop.ndim > 2 else face_crop
            )

            face_crop = cv2.resize(
                face_crop,
                self.final_image_size[::-1],
                interpolation=self.opencv_interpolation,
            )

            face_crop = opencvbgr_to_bob(np.array(face_crop))

        else:

            # Expanding the borders
            top_expanded = int(np.maximum(top - self.margin * height, 0))
            left_expanded = int(np.maximum(left - self.margin * width, 0))

            bottom_expanded = int(
                np.minimum(bottom + self.margin * height, X.shape[1])
            )
            right_expanded = int(
                np.minimum(right + self.margin * width, X.shape[2])
            )

            face_crop = X[
                :,
                top_expanded:bottom_expanded,
                left_expanded:right_expanded,
            ]

        return face_crop

    def _more_tags(self):
        return {"requires_fit": False}

    def fit(self, X, y=None):
        return self
