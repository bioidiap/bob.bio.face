from bob.bio.face.database import RFWDatabase

# In case protocol is comming from chain loading
# https://www.idiap.ch/software/bob/docs/bob/bob.extension/stable/py_api.html#bob.extension.config.load
if "protocol" not in locals():
    protocol = "original"


database = RFWDatabase(protocol=protocol)
