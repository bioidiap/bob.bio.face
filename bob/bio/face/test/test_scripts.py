import os
import shutil
import tempfile

import pkg_resources

from click.testing import CliRunner


def test_display_annotations():

    from bob.bio.face.script.display_face_annotations import (
        display_face_annotations,
    )

    try:
        tmp_dir = tempfile.mkdtemp(prefix="bobtest_")
        annotations_dir = pkg_resources.resource_filename(
            "bob.bio.face.test", "data/annotations/"
        )
        runner = CliRunner()
        result = runner.invoke(
            display_face_annotations,
            args=(
                "--database",
                "dummy",
                "--groups",
                "train",
                "--groups",
                "dev",
                "--annotations-dir",
                annotations_dir,
                "--output-dir",
                tmp_dir,
                "--keep-all",
                "--self-test",
            ),
        )
        assertion_error_message = (
            "Command exited with this output: `{}' \n"
            "If the output is empty, you can run this script locally to see "
            "what is wrong:\n"
            "$ bob bio display-face-annotations -vvv -d dummy -g train -g dev -a ./annotations/ -o /tmp/temp_annotated"
            "".format(result.output)
        )
        assert result.exit_code == 0, assertion_error_message

        # Checks if an annotated sample exists
        sample_1_path = os.path.join(tmp_dir, "s1", "1.png")
        assertion_error_message = "File '{}' not created.".format(sample_1_path)
        assert os.path.isfile(sample_1_path), assertion_error_message

    finally:
        shutil.rmtree(tmp_dir)
