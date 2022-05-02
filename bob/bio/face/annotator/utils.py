import math

available_sources = {
    "direct": ("topleft", "bottomright"),
    "eyes": ("leye", "reye"),
    "left-profile": ("eye", "mouth"),
    "right-profile": ("eye", "mouth"),
    "ellipse": ("center", "angle", "axis_radius"),
}

# This struct specifies, which paddings should be applied to which source.
# All values are relative to the inter-node distance
default_paddings = {
    "direct": None,
    "eyes": {
        "left": -1.0,
        "right": +1.0,
        "top": -0.7,
        "bottom": 1.7,
    },  # These parameters are used to match Cosmin's implementation (which was buggy...)
    "left-profile": {"left": -0.2, "right": +0.8, "top": -1.0, "bottom": 1.0},
    "right-profile": {"left": -0.8, "right": +0.2, "top": -1.0, "bottom": 1.0},
    "ellipse": None,
}


def _to_int(value):
    """Converts a value to int by rounding"""
    if isinstance(value, tuple):
        return tuple(map(_to_int, value))
    return int(round(value))


class BoundingBox:
    """A bounding box class storing top, left, height and width of an rectangle."""

    def __init__(self, topleft: tuple, size: tuple = None, **kwargs):
        """Creates a new BoundingBox
        Parameters
        ----------
        topleft
            The top left corner of the bounding box as (y, x) tuple
        size
            The size of the bounding box as (height, width) tuple
        """
        super().__init__(**kwargs)
        if isinstance(topleft, BoundingBox):
            self.__init__(topleft.topleft, topleft.size)
            return

        if topleft is None:
            raise ValueError(
                "BoundingBox must be initialized with a topleft and a size"
            )
        self._topleft = tuple(topleft)
        if size is None:
            raise ValueError("BoundingBox needs a size")
        self._size = tuple(size)

    @property
    def topleft_f(self):
        """The top-left position of the bounding box as floating point values, read access only"""
        return self._topleft

    @property
    def topleft(self):
        """The top-left position of the bounding box as integral values, read access only"""
        return _to_int(self.topleft_f)

    @property
    def size_f(self):
        """The size of the bounding box as floating point values, read access only"""
        return self._size

    @property
    def size(self):
        """The size of the bounding box as integral values, read access only"""
        return _to_int(self.size_f)

    @property
    def top_f(self):
        """The top position of the bounding box as floating point values, read access only"""
        return self._topleft[0]

    @property
    def top(self):
        """The top position of the bounding box as integral values, read access only"""
        return _to_int(self.top_f)

    @property
    def left_f(self):
        """The left position of the bounding box as floating point values, read access only"""
        return self._topleft[1]

    @property
    def left(self):
        """The left position of the bounding box as integral values, read access only"""
        return _to_int(self.left_f)

    @property
    def height_f(self):
        """The height of the bounding box as floating point values, read access only"""
        return self._size[0]

    @property
    def height(self):
        """The height of the bounding box as integral values, read access only"""
        return _to_int(self.height_f)

    @property
    def width_f(self):
        """The width of the bounding box as floating point values, read access only"""
        return self._size[1]

    @property
    def width(self):
        """The width of the bounding box as integral values, read access only"""
        return _to_int(self.width_f)

    @property
    def right_f(self):
        """The right position of the bounding box as floating point values, read access only"""
        return self.left_f + self.width_f

    @property
    def right(self):
        """The right position of the bounding box as integral values, read access only"""
        return _to_int(self.right_f)

    @property
    def bottom_f(self):
        """The bottom position of the bounding box as floating point values, read access only"""
        return self.top_f + self.height_f

    @property
    def bottom(self):
        """The bottom position of the bounding box as integral values, read access only"""
        return _to_int(self.bottom_f)

    @property
    def bottomright_f(self):
        """The bottom right corner of the bounding box as floating point values, read access only"""
        return (self.bottom_f, self.right_f)

    @property
    def bottomright(self):
        """The bottom right corner of the bounding box as integral values, read access only"""
        return _to_int(self.bottomright_f)

    @property
    def center(self):
        """The center of the bounding box, read access only"""
        return (self.top_f + self.bottom_f) // 2, (
            self.left_f + self.right_f
        ) // 2

    @property
    def area(self):
        """The area (height x width) of the bounding box, read access only"""
        return self.height_f * self.width_f

    def contains(self, point):
        """Returns True if the given point is inside the bounding box
        Parameters
        ----------
        point : tuple
            A point as (x, y) tuple
        Returns
        -------
        bool
            True if the point is inside the bounding box
        """
        return (
            self.top_f <= point[0] < self.bottom_f
            and self.left_f <= point[1] < self.right_f
        )

    def is_valid_for(self, size: tuple) -> bool:
        """Checks if the bounding box is inside the given image size
        Parameters
        ----------
        size
            The size of the image to testA size as (height, width) tuple
        Returns
        -------
        bool
            True if the bounding box is inside the image boundaries
        """
        return (
            self.top_f >= 0
            and self.left_f >= 0
            and self.bottom_f <= size[0]
            and self.right_f <= size[1]
        )

    def mirror_x(self, width: int) -> "BoundingBox":
        """Returns a horizontally mirrored version of this BoundingBox
        Parameters
        ----------
        width
            The width of the image at which this bounding box should be mirrored
        Returns
        -------
        bounding_box
            The mirrored version of this bounding box
        """
        return BoundingBox((self.top_f, width - self.right_f), self.size_f)

    def overlap(self, other: "BoundingBox") -> "BoundingBox":
        """Returns the overlapping bounding box between this and the given bounding box
        Parameters
        ----------
        other
            The other bounding box to compute the overlap with
        Returns
        -------
        bounding_box
            The overlap between this and the other bounding box
        """
        if self.top_f > other.bottom_f or other.top_f > self.bottom_f:
            return BoundingBox((0, 0), (0, 0))
        if self.left_f > other.right_f or other.left_f > self.right_f:
            return BoundingBox((0, 0), (0, 0))
        max_top = max(self.top_f, other.top_f)
        max_left = max(self.left_f, other.left_f)
        min_bottom = min(self.bottom_f, other.bottom_f)
        min_right = min(self.right_f, other.right_f)
        return BoundingBox(
            (
                max_top,
                max_left,
            ),
            (
                min_bottom - max_top,
                min_right - max_left,
            ),
        )

    def scale(self, scale: float, centered=False) -> "BoundingBox":
        """Returns a scaled version of this BoundingBox
        When the centered parameter is set to True, the transformation center will be in the center of this bounding box, otherwise it will be at (0,0)
        Parameters
        ----------
        scale
            The scale with which this bounding box should be shifted
        centered
            Should the scaling done with repect to the center of the bounding box?
        Returns
        -------
        bounding_box
            The scaled version of this bounding box
        """
        if centered:
            return BoundingBox(
                (
                    self.top_f - self.height_f / 2 * (scale - 1),
                    self.left_f - self.width_f / 2 * (scale - 1),
                ),
                (self.height_f * scale, self.width_f * scale),
            )
        else:
            return BoundingBox(
                (self.top_f * scale, self.left_f * scale),
                (self.height_f * scale, self.width_f * scale),
            )

    def shift(self, offset: tuple) -> "BoundingBox":
        """Returns a shifted version of this BoundingBox
        Parameters
        ----------
        offset
            The offset with which this bounding box should be shifted
        Returns
        -------
        bounding_box
            The shifted version of this bounding box
        """
        return BoundingBox(
            (self.top_f + offset[0], self.left_f + offset[1]), self.size_f
        )

    def similarity(self, other: "BoundingBox") -> float:
        """Returns the Jaccard similarity index between this and the given BoundingBox
        The Jaccard similarity coefficient between two bounding boxes is defined as their intersection divided by their union.
        Parameters
        ----------
        other
            The other bounding box to compute the overlap with
        Returns
        -------
        sim : float
            The Jaccard similarity index between this and the given BoundingBox
        """
        max_top = max(self.top_f, other.top_f)
        max_left = max(self.left_f, other.left_f)
        min_bottom = min(self.bottom_f, other.bottom_f)
        min_right = min(self.right_f, other.right_f)

        # no overlap?
        if max_left >= min_right or max_top >= min_bottom:
            return 0.0

        # compute overlap
        intersection = (min_bottom - max_top) * (min_right - max_left)
        return intersection / (self.area + other.area - intersection)

    def __eq__(self, other: object) -> bool:
        return self.topleft_f == other.topleft_f and self.size_f == other.size_f


def bounding_box_from_annotation(source=None, padding=None, **kwargs):
    """bounding_box_from_annotation(source, padding, **kwargs) -> bounding_box

    Creates a bounding box from the given parameters, which are, in general, annotations read using :py:func:`bob.bio.base.utils.annotations.read_annotation_file`.
    Different kinds of annotations are supported, given by the ``source`` keyword:

    * ``direct`` : bounding boxes are directly specified by keyword arguments ``topleft`` and ``bottomright``
    * ``eyes`` : the left and right eyes are specified by keyword arguments ``leye`` and ``reye``
    * ``left-profile`` : the left eye and the mouth are specified by keyword arguments ``eye`` and ``mouth``
    * ``right-profile`` : the right eye and the mouth are specified by keyword arguments ``eye`` and ``mouth``
    * ``ellipse`` : the face ellipse as well as face angle and axis radius is provided by keyword arguments ``center``, ``angle`` and ``axis_radius``

    If a ``source`` is specified, the according keywords must be given as well.
    Otherwise, the source is estimated from the given keyword parameters if possible.

    If 'topleft' and 'bottomright' are given (i.e., the 'direct' source), they are taken as is.
    Note that the 'bottomright' is NOT included in the bounding box.
    Please assure that the aspect ratio of the bounding box is 6:5 (height : width).

    For source 'ellipse', the bounding box is computed to capture the whole ellipse, even if it is rotated.

    For other sources (i.e., 'eyes'), the center of the two given positions is computed, and the ``padding`` is applied, which is relative to the distance between the two given points.
    If ``padding`` is ``None`` (the default) the default_paddings of this source are used instead.
    These padding is required to keep an aspect ratio of 6:5.

    Parameters
    ----------

    source : str or ``None``
      The type of annotations present in the list of keyword arguments, see above.

    padding : {'top':float, 'bottom':float, 'left':float, 'right':float}
      This padding is added to the center between the given points, to define the top left and bottom right positions in the bounding box; values are relative to the distance between the two given points; ignored for some of the ``source``\\s

    kwargs : key=value
      Further keyword arguments specifying the annotations.

    Returns
    -------

    bounding_box : :py:class:`BoundingBox`
      The bounding box that was estimated from the given annotations.
    """

    if source is None:
        # try to estimate the source
        for s, k in available_sources.items():
            # check if the according keyword arguments are given
            if k[0] in kwargs and k[1] in kwargs:
                # check if we already assigned a source before
                if source is not None:
                    raise ValueError(
                        "The given list of keywords (%s) is ambiguous. Please specify a source"
                        % kwargs
                    )
                # assign source
                source = s

        # check if a source could be estimated from the keywords
        if source is None:
            raise ValueError(
                "The given list of keywords (%s) could not be interpreted"
                % kwargs
            )

    assert source in available_sources

    # use default padding if not specified
    if padding is None:
        padding = default_paddings[source]

    keys = available_sources[source]
    if source == "ellipse":
        # compute the tight bounding box for the ellipse
        angle = kwargs["angle"]
        axis = kwargs["axis_radius"]
        center = kwargs["center"]
        dx = abs(math.cos(angle) * axis[0]) + abs(math.sin(angle) * axis[1])
        dy = abs(math.sin(angle) * axis[0]) + abs(math.cos(angle) * axis[1])
        top = center[0] - dy
        bottom = center[0] + dy
        left = center[1] - dx
        right = center[1] + dx
    elif padding is None:
        # There is no padding to be applied -> take nodes as they are
        top = kwargs[keys[0]][0]
        bottom = kwargs[keys[1]][0]
        left = kwargs[keys[0]][1]
        right = kwargs[keys[1]][1]
    else:
        # apply padding
        pos_0 = kwargs[keys[0]]
        pos_1 = kwargs[keys[1]]
        tb_center = float(pos_0[0] + pos_1[0]) / 2.0
        lr_center = float(pos_0[1] + pos_1[1]) / 2.0
        distance = math.sqrt(
            (pos_0[0] - pos_1[0]) ** 2 + (pos_0[1] - pos_1[1]) ** 2
        )

        top = tb_center + padding["top"] * distance
        bottom = tb_center + padding["bottom"] * distance
        left = lr_center + padding["left"] * distance
        right = lr_center + padding["right"] * distance

    return BoundingBox((top, left), (bottom - top, right - left))


def expected_eye_positions(bounding_box, padding=None):
    """expected_eye_positions(bounding_box, padding) -> eyes

    Computes the expected eye positions based on the relative coordinates of the bounding box.

    This function can be used to translate between bounding-box-based image cropping and eye-location-based alignment.
    The returned eye locations return the **average** eye locations, no landmark detection is performed.

    **Parameters:**

    ``bounding_box`` : :py:class:`BoundingBox`
      The face bounding box.

    ``padding`` : {'top':float, 'bottom':float, 'left':float, 'right':float}
      The padding that was used for the ``eyes`` source in :py:func:`bounding_box_from_annotation`, has a proper default.

    **Returns:**

    ``eyes`` : {'reye' : (rey, rex), 'leye' : (ley, lex)}
      A dictionary containing the average left and right eye annotation.
    """
    if padding is None:
        padding = default_paddings["eyes"]
    top, left, right = padding["top"], padding["left"], padding["right"]
    inter_eye_distance = (bounding_box.size[1]) / (right - left)
    return {
        "reye": (
            bounding_box.top_f - top * inter_eye_distance,
            bounding_box.left_f - left / 2.0 * inter_eye_distance,
        ),
        "leye": (
            bounding_box.top_f - top * inter_eye_distance,
            bounding_box.right_f - right / 2.0 * inter_eye_distance,
        ),
    }


def bounding_box_to_annotations(bbx):
    """Converts :any:`BoundingBox` to dictionary annotations.

    Parameters
    ----------
    bbx : :any:`BoundingBox`
        The given bounding box.

    Returns
    -------
    dict
        A dictionary with topleft and bottomright keys.
    """
    landmarks = {
        "topleft": bbx.topleft,
        "bottomright": bbx.bottomright,
    }
    return landmarks


def min_face_size_validator(annotations, min_face_size=(32, 32)):
    """Validates annotations based on face's minimal size.

    Parameters
    ----------
    annotations : dict
        The annotations in dictionary format.
    min_face_size : (:obj:`int`, :obj:`int`), optional
        The minimal size of a face.

    Returns
    -------
    bool
        True, if the face is large enough.
    """
    if not annotations:
        return False
    for source in ("direct", "eyes", None):
        try:
            bbx = bounding_box_from_annotation(source=source, **annotations)
            break
        except Exception:
            if source is None:
                raise
            else:
                pass
    if bbx.size[0] < min_face_size[0] or bbx.size[1] < min_face_size[1]:
        return False
    return True
