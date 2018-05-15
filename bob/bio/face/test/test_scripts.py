import bob.bio.base.test.utils
import bob.bio.face

def test_display_annotations():

  from bob.bio.face.script.display_face_annotations import main

  with bob.bio.base.test.utils.Quiet():
    parameters = ['-d', 'dummy', '-a', '/very/unlikely/directory', '--self-test']
    main(parameters)
