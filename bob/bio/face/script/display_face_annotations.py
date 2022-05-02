#!../bin/python

"""This script displays the images with annotations provided by any face database.

By default, all images and their corresponding annotations are displayed, and you have to press ``Enter`` after each image.
If the database does not include annotations, or you want to display a different set of annotations, you can specify the ``--annotation-directory`` (and if required modify the ``--annotation-file-extension`` and ``--annotation-file-type``.
The list of images can be narrowed down by the ``--file-ids`` parameter, where the file ids are specific to the database.

Note that this script can only be used with face image databases, not with video or other biometric databases.
"""

from __future__ import print_function

import logging
import os

import click

from bob.bio.base.utils.annotations import read_annotation_file
from bob.bio.face.color import gray_to_rgb
from bob.extension.scripts.click_helper import (
    ConfigCommand,
    ResourceOption,
    verbosity_option,
)

logger = logging.getLogger(__name__)


@click.command(
    entry_point_group="bob.bio.config",
    cls=ConfigCommand,
    epilog="""\b
Examples:

    $ bob bio display-face-annotations -vvv -d <database> -a <annot_dir>
""",
)
@click.option(
    "-d",
    "--database",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.bio.database",
    help="Select the database for which the images plus annotations should be shown.",
)
@click.option(
    "-V",
    "--video",
    "is_video",
    is_flag=True,
    help="Provide this flag if your database is a video database. "
    "For video databases, the annotations for the first frame is shown.",
)
@click.option(
    "-a",
    "--annotations-dir",
    help="Use the annotations stored in this directory "
    "(when annotated with `$ bob bio annnotate` for example). "
    "If not given, will try to load the annotations from the database.",
)
@click.option(
    "-x",
    "--annotations-extension",
    default=".json",
    show_default=True,
    help="Annotations files have the given filename extension.",
)
@click.option(
    "-t",
    "--annotations-type",
    default="json",
    show_default=True,
    help="Annotations type given to bob.bio.base.read_annotations.",
)
@click.option(
    "-n",
    "--display-names",
    is_flag=True,
    help="Plot the names of the annotations, too.",
)
@click.option(
    "-m",
    "--marker-style",
    default="rx",
    show_default=True,
    help="Select the marker style",
)
@click.option(
    "-M",
    "--marker-size",
    type=float,
    default=10.0,
    show_default=True,
    help="Select the marker size",
)
@click.option(
    "-C",
    "--font-color",
    default="b",
    show_default=True,
    help="Select the color for the annotations names",
)
@click.option(
    "-F",
    "--font-size",
    type=int,
    default=16,
    show_default=True,
    help="Select the font size for the annotations names",
)
@click.option(
    "-o",
    "--output-dir",
    help="If given, it will save the plots in this output file instead of showing them. This option is useful when you don't have a display server (ssh).",
)
@click.option(
    "-k",
    "--keep-all",
    is_flag=True,
    help="When -o is given: keeps every annotated samples instead of just one.",
)
@click.option(
    "--self-test",
    is_flag=True,
    help="Prevents outputing to the screen and waiting for user input.",
)
@click.option(
    "--groups",
    "-g",
    multiple=True,
    default=["dev", "eval"],
    show_default=True,
    help="Biometric Database group that will be displayed.",
)
@verbosity_option(cls=ResourceOption)
def display_face_annotations(
    database,
    is_video,
    annotations_dir,
    annotations_extension,
    annotations_type,
    marker_style,
    marker_size,
    display_names,
    font_color,
    font_size,
    output_dir,
    keep_all,
    self_test,
    groups,
    **kwargs,
):
    """
    Plots annotations on the corresponding face picture.
    """
    logger.debug("Retrieving samples from database.")
    samples = database.all_samples(groups)

    logger.debug(f"{len(samples)} samples loaded from database.")

    # open figure
    from matplotlib import pyplot

    if not self_test and not output_dir:
        pyplot.ion()
        pyplot.show()
    else:
        pyplot.ioff()
    pyplot.figure()

    for sample in samples:
        # load image
        logger.info("loading image for sample %s", sample.key)
        image = sample.data
        if is_video:
            frame_id, image, _ = image[0]
        # convert to color if it is not
        if image.ndim == 2:
            image = gray_to_rgb(image)

        # get annotations
        annotations = {}
        if annotations_dir is not None:
            # Loads the corresponding annotations file
            annotations_file = os.path.join(
                annotations_dir, sample.key + annotations_extension
            )
            if os.path.exists(annotations_file):
                logger.info(
                    "Loading annotations from file %s", annotations_file
                )
                annotations = read_annotation_file(
                    annotations_file, annotations_type
                )
            else:
                logger.warn(
                    "Could not find annotation file %s", annotations_file
                )
        else:
            # get annotations from database
            annotations = database.annotations(sample)

        if not annotations:
            logger.warn("Could not find annotations for file %s", sample.key)
            continue

        if is_video:
            assert frame_id in annotations, annotations
            annotations = annotations[frame_id]

        pyplot.clf()
        pyplot.imshow(image.transpose(1, 2, 0))

        global_annotation = []
        for n, a in annotations.items():
            if isinstance(a, (list, tuple)) and len(a) == 2:
                pyplot.plot(
                    a[1],
                    a[0],
                    marker_style,
                    ms=marker_size,
                    mew=marker_size / 5.0,
                )
                if display_names:
                    pyplot.annotate(
                        n, (a[1], a[0]), color=font_color, fontsize=font_size
                    )
            else:
                global_annotation.append("%s=%s" % (n, a))

        # plot all global annotations, at the top center of the image
        pyplot.annotate(
            ";".join(global_annotation),
            (image.shape[-1] / 2, 0),
            color=font_color,
            fontsize=font_size,
            ha="center",
            va="baseline",
        )

        pyplot.gca().set_aspect("equal")
        pyplot.gca().autoscale(tight=True)

        if output_dir is None:
            if self_test:
                raise RuntimeError("Do not run self_test without --output_dir.")
            pyplot.pause(0.001)
        else:
            if keep_all:
                output_path = os.path.join(output_dir, sample.key + ".png")
            else:
                output_path = os.path.join(output_dir, "annotated.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pyplot.savefig(output_path)

        if not self_test:
            input_text = (
                "Press Enter to continue to the next image (or Ctrl-C to exit)"
            )
            input(input_text)
