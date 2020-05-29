from bob.extension.config import load
import pkg_resources
import numpy as np
from bob.pipelines import Sample, SampleSet


def get_fake_sample(face_size=(160, 160), eyes={"leye": (46, 107), "reye": (46, 53)}):

    data = np.random.rand(3,400,400)
    annotations={"leye": (115, 267), "reye": (115, 132)}
    return Sample(data, key="1", annotations=annotations)




def test_facenet_pipeline():
    config_name = pkg_resources.resource_filename('bob.bio.face', 'config/transformers/eyes_crop/facenet.py')
    transformer = load([config_name]).transformer
    
    #import ipdb; ipdb.set_trace()
    fake_sample = get_fake_sample()

    #transformed_sample = transformer.transform([fake_sample])[0].data

    #import ipdb; ipdb.set_trace()    
    transformed_sample = transformer.transform([fake_sample])[0]
    assert transformed_sample.data.size == 160

    pass