#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Yu Linghu & Xinyi Zhang <yu.linghu@uzh.ch, xinyi.zhang@uzh.ch>
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
from bob.extension.download import get_file
import numpy as np
import os


class MxNetTransformer(TransformerMixin, BaseEstimator):

    """
    Base Transformer for MxNet architectures.

    Parameters:
    -----------

      checkpoint_path : str
         Path containing the checkpoint

      config : str
         json file containing the DNN spec

      use_gpu: bool
    """

    def __init__(
        self,
        checkpoint_path=None,
        config=None,
        use_gpu=False,
        memory_demanding=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.use_gpu = use_gpu
        self.model = None
        self.memory_demanding = memory_demanding

    def _load_model(self):
        import mxnet as mx
        from mxnet import gluon
        import warnings

        ctx = mx.gpu() if self.use_gpu else mx.cpu()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            deserialized_net = gluon.nn.SymbolBlock.imports(
                self.config, ["data"], self.checkpoint_path, ctx=ctx
            )

        self.model = deserialized_net

    def transform(self, X):

        import mxnet as mx

        if self.model is None:
            self._load_model()

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


class ArcFaceInsightFace_LResNet100(MxNetTransformer):
    """
    Extracts features using deep face recognition models under MxNet Interfaces.
  
    Users can download the pretrained face recognition models with MxNet Interface. The path to downloaded models (and weights) should be specified while calling this class, usually in the configuration file of an experiment. 
 
    Examples: (Pretrained ResNet models): `LResNet100E-IR,ArcFace@ms1m-refine-v2 <https://github.com/deepinsight/insightface>`_  
  
    The extracted features can be combined with different the algorithms.  

    """

    def __init__(self, memory_demanding=False, use_gpu=False):
        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/mxnet/arcface_r100_v1_mxnet.tar.gz",
            "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/mxnet/arcface_r100_v1_mxnet.tar.gz",
        ]
        filename = get_file(
            "arcface_r100_v1_mxnet.tar.gz",
            urls,
            cache_subdir="data/mxnet/arcface_r100_v1_mxnet",
            file_hash="050ce7d6e731e560127c705f61391f48",
            extract=True,
        )
        path = os.path.dirname(filename)
        checkpoint_path = os.path.join(path, "model-symbol.json")
        config = os.path.join(path, "model-0000.params")

        super(ArcFaceInsightFace_LResNet100, self).__init__(
            checkpoint_path=checkpoint_path,
            config=config,
            use_gpu=use_gpu,
            memory_demanding=memory_demanding,
        )

    def _load_model(self):
        import mxnet as mx

        sym, arg_params, aux_params = mx.model.load_checkpoint(
            os.path.join(os.path.dirname(self.checkpoint_path), "model"), 0
        )

        all_layers = sym.get_internals()
        sym = all_layers["fc1_output"]

        # LOADING CHECKPOINT
        ctx = mx.gpu() if self.use_gpu else mx.cpu()
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        data_shape = (1, 3, 112, 112)
        model.bind(data_shapes=[("data", data_shape)])
        model.set_params(arg_params, aux_params)

        self.model = model
