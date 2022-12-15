from bob.bio.face.database import CBSRNirVis2Database

# In case protocol is comming from chain loading
# https://www.idiap.ch/software/bob/docs/bob/bob.extension/stable/py_api.html#bob.extension.config.load
if "protocol" not in locals():
    protocol = "view2_1"


database = CBSRNirVis2Database(protocol=protocol)
