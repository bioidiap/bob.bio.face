from bob.bio.face.pytorch_datasets.webface42m import WebFace42M
from bob.extension import rc

import pytest


@pytest.mark.skipif(
    rc.get("bob.bio.face.webface42M.directory") is None,
    reason="WEBFace42M  not available. Please do `bob config set bob.bio.face.ijbc.directory [IJBC PATH]` to set the IJBC data path.",
)
def test_webface42M():

    dataset = WebFace42M()

    sample = dataset[0]
    assert sample["label"] == 0
    assert sample["data"].shape == (3, 112, 112)

    sample = dataset[100000]
    assert sample["label"] == 4960
    assert sample["data"].shape == (3, 112, 112)

    sample = dataset[42474557]
    assert sample["label"] == 2059905
    assert sample["data"].shape == (3, 112, 112)

    sample = dataset[-1]
    assert sample["label"] == 2059905
    assert sample["data"].shape == (3, 112, 112)

