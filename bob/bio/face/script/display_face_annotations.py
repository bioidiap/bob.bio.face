#!../bin/python

"""This script displays the iamges with annotations provided by any face database.
Basically, anything that can be used as a --database for verify.py can be specified here as well, including configuration files and ``database`` resources: ``resources.py -d database``.

By default, all images and their corresponding annotations are displayed, and you have to press ``Enter`` after each image.
If the database does not include annotations, or you want to display a different set of annotations, you can specify the ``--annotation-directory`` (and if required modify the ``--annotation-file-extension`` and ``--annotation-file-type``.
The list of images can be narrowed down by the ``--file-ids`` parameter, where the file ids are specific to the database.

Note that this script can only be used with face image databases, not with video or other biometric databases.
"""

from __future__ import print_function

import os
import argparse
import sys

import bob.bio.base

import bob.core
logger = bob.core.log.setup("bob.bio.face")

def command_line_arguments(command_line_parameters):
  """Defines the command line parameters that are accepted."""

  # create parser
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # add parameters
  parser.add_argument('-d', '--database', nargs = '+', help = 'Select the database for which the images plus annotations should be shown.')
  parser.add_argument('-V', '--video', action = 'store_true', help = 'Provide this flag if your database is a video database. For video databases, the annotations for the first frame is shown.')
  parser.add_argument('-f', '--file-ids', nargs = '+', help = 'If given, only the images of the --database with the given file id are shown (non-existing IDs will be silently skipped).')
  parser.add_argument('-a', '--annotation-directory', help = 'If given, use the annotations stored in the given annotation directory (this might be required for some databases).')
  parser.add_argument('-x', '--annotation-file-extension', default = '.pos', help = 'Annotation files have the given filename extension.')
  parser.add_argument('-t', '--annotation-file-type', default = 'named', help = 'Select the annotation file style, see bob.db.base documentation for valid types.')
  parser.add_argument('-n', '--annotate-names', action = 'store_true', help = 'Plot the names of the annotations, too.')
  parser.add_argument('-m', '--marker-style', default='rx', help = 'Select the marker style')
  parser.add_argument('-M', '--marker-size', type=float, default=10., help = 'Select the marker size')
  parser.add_argument('-F', '--font-size', type=int, default=16, help = 'Select the font size for the annotation names')
  parser.add_argument('-C', '--font-color', default = 'b', help = 'Select the color for the annotation names')
  parser.add_argument('--database-directories-file', metavar = 'FILE', default = "%s/.bob_bio_databases.txt" % os.environ["HOME"], help = 'An optional file, where database directories are stored (to avoid changing the database configurations)')
  parser.add_argument('-o', '--output', help = "If given, it will save the plots in this output file instead of showing them. This option is useful when you don't have a display server (ssh).")

  parser.add_argument('--self-test', action='store_true', help=argparse.SUPPRESS)

  bob.core.log.add_command_line_option(parser)
  args = parser.parse_args(command_line_parameters)
  bob.core.log.set_verbosity_level(logger, args.verbose)

  return args


def main(command_line_parameters=None):

  args = command_line_arguments(command_line_parameters)

  # load database
  database = bob.bio.base.load_resource("".join(args.database), "database")
  # replace directories
  if isinstance(database, bob.bio.base.database.BioDatabase):
    database.replace_directories(args.database_directories_file)

  # get all files
  if bob.bio.base.utils.is_argument_available('flat', database.all_files):
    files = database.all_files(flat=True)
  else:
    files = database.all_files()

  # filter file ids; convert them to str first
  if args.file_ids is not None:
    files = [f for f in files if str(f.id) in args.file_ids]

  # open figure
  if not args.self_test:
    from matplotlib import pyplot
    if args.output is None:
      pyplot.ion()
      pyplot.show()
    else:
      pyplot.ioff()
    pyplot.figure()

  for f in files:
    # load image
    logger.info("loading image for file %s", f.id)
    image = f.load(database.original_directory, database.original_extension)
    if args.video:
      frame_id, image, _ = image[0]
    # convert to color if it is not
    if image.ndim == 2:
      image = bob.ip.color.gray_to_rgb(image)

    # get annotations
    annotations = {}
    if args.annotation_directory is not None:
      # load annotation file
      annotation_file = f.make_path(args.annotation_directory, args.annotation_file_extension)
      if os.path.exists(annotation_file):
        logger.info("Loading annotations from file %s", annotation_file)
        annotations = bob.db.base.read_annotation_file(annotation_file, args.annotation_file_type)
      else:
        logger.warn("Could not find annotation file %s", annotation_file)
    else:
      # get annotations from database
      annotations = database.annotations(f)

    if not annotations:
      logger.warn("Could not find annotations for file %s", f.id)

    if args.video:
      assert frame_id in annotations, annotations
      annotations = annotations[frame_id]

    if not args.self_test:
      pyplot.clf()
      pyplot.imshow(image.transpose(1,2,0))

      global_annotation = []
      for n,a in annotations.items():
        if isinstance(a, (list,tuple)) and len(a) == 2:
          pyplot.plot(a[1], a[0], args.marker_style, ms = args.marker_size, mew = args.marker_size / 5.)
          if args.annotate_names:
            pyplot.annotate(n, (a[1], a[0]), color=args.font_color, fontsize=args.font_size)
        else:
          global_annotation.append("%s=%s"%(n,a))

      # plot all global annotations, at the top center of the image
      pyplot.annotate(";".join(global_annotation), (image.shape[-1]/2, 0), color=args.font_color, fontsize=args.font_size, ha='center', va='baseline')

      pyplot.gca().set_aspect("equal")
      pyplot.gca().autoscale(tight=True)
      if args.output is None:
        pyplot.pause(0.001)
      else:
        pyplot.savefig(args.output)

      input_text = "Press Enter to continue to the next image (or Ctrl-C + Enter to exit)"
      if sys.version_info >= (3, 0):
        input(input_text)
      else:
        raw_input(input_text)
