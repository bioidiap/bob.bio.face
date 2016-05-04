import bob.bio.base.test.utils
import bob.bio.face

@bob.bio.base.test.utils.grid_available
def test_baselines():
  # test that all of the baselines would execute
  from bob.bio.face.script.baselines import available_databases, all_algorithms, main

  with bob.bio.base.test.utils.Quiet():
    for database in available_databases:
      parameters = ['-d', database, '--dry-run']
      main(parameters)
      parameters.append('--grid')
      main(parameters)
      parameters.extend(['-e', 'HTER'])
      main(parameters)

    for algorithm in all_algorithms:
      parameters = ['-a', algorithm, '--dry-run']
      main(parameters)
      parameters.append('-g')
      main(parameters)
      parameters.extend(['-e', 'HTER'])
      main(parameters)

def test_display_annotations():

  from bob.bio.face.script.display_face_annotations import main

  with bob.bio.base.test.utils.Quiet():
    parameters = ['-d', 'dummy', '-a', '/very/unlikely/directory', '--self-test']
    main(parameters)
