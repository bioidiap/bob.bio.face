"""
Load and predict using checkpoints based on mxnet
"""

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
import numpy as np
from bob.bio.face.embeddings import download_model
import pkg_resources
import os
from bob.extension import rc


class ArcFaceInsightFace(TransformerMixin, BaseEstimator):
    """
    ArcFace from Insight Face.

    Model and source code taken from the repository

    https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/face_recognition.py

    """

    def __init__(self, use_gpu=False, memory_demanding=False, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.use_gpu = use_gpu
        self.memory_demanding = memory_demanding

        internal_path = pkg_resources.resource_filename(
            __name__, os.path.join("data", "arcface_insightface"),
        )

        checkpoint_path = (
            internal_path
            if rc["bob.bio.face.models.ArcFaceInsightFace"] is None
            else rc["bob.bio.face.models.ArcFaceInsightFace"]
        )

        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/mxnet/arcface_r100_v1_mxnet.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/mxnet/arcface_r100_v1_mxnet.tar.gz"
        ]

        download_model(checkpoint_path, urls, "arcface_r100_v1_mxnet.tar.gz")

        self.checkpoint_path = checkpoint_path

    def load_model(self):
        import mxnet as mx

        sym, arg_params, aux_params = mx.model.load_checkpoint(
            os.path.join(self.checkpoint_path, "model"), 0
        )

        all_layers = sym.get_internals()
        sym = all_layers["fc1_output"]

        # LOADING CHECKPOINT
        ctx = mx.gpu() if self.use_gpu else mx.cpu()
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        data_shape = (1, 3, 112, 112)
        model.bind(data_shapes=[("data", data_shape)])
        model.set_params(arg_params, aux_params)

        # warmup
        data = mx.nd.zeros(shape=data_shape)
        db = mx.io.DataBatch(data=(data,))
        model.forward(db, is_train=False)
        embedding = model.get_outputs()[0].asnumpy()
        self.model = model

    def transform(self, X):
        import mxnet as mx

        if self.model is None:
            self.load_model()

        X = check_array(X, allow_nd=True)

        def _transform(X):
            X = mx.nd.array(X)
            db = mx.io.DataBatch(data=(X,))
            self.model.forward(db, is_train=False)
            return self.model.get_outputs()[0].asnumpy()

        if self.memory_demanding:
            return np.array([_transform(x[None, ...]) for x in X])
        else:
            return _transform(X)

    def __getstate__(self):
        # Handling unpicklable objects
        d = self.__dict__.copy()
        d["model"] = None
        return d

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}
