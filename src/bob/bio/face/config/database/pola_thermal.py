from bob.bio.face.database import PolaThermalDatabase

# In case protocol is comming from chain loading
# https://www.idiap.ch/software/bob/docs/bob/bob.extension/stable/py_api.html#bob.extension.config.load
if "protocol" not in locals():
    protocol = "VIS-thermal-overall-split1"


database = PolaThermalDatabase(protocol=protocol)
