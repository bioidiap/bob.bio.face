from bob.bio.face.database import MEDSDatabase

# In case protocol is comming from chain loading
# https://www.idiap.ch/software/bob/docs/bob/bob.extension/stable/py_api.html#bob.extension.config.load
if "protocol" not in locals():
    protocol = "verification_fold1"


database = MEDSDatabase(protocol=protocol)
