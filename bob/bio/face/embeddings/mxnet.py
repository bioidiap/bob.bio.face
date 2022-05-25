#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Yu Linghu & Xinyi Zhang <yu.linghu@uzh.ch, xinyi.zhang@uzh.ch>
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import os

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

from bob.bio.face.annotator import MTCNN
from bob.extension.download import get_file


class MxNetTransformer(TransformerMixin, BaseEstimator):

    """
    Base Transformer for MxNet architectures.

    Parameters:
    -----------

      checkpoint_path : str
         Path containing the checkpoint

      config : str
         json file containing the DNN spec

      preprocessor:
         A function that will transform the data right before forward. The default transformation is `X=X`

      use_gpu: bool
    """

    def __init__(
        self,
        checkpoint_path=None,
        config=None,
        use_gpu=False,
        memory_demanding=False,
        preprocessor=lambda x: x,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.use_gpu = use_gpu
        self.model = None
        self.memory_demanding = memory_demanding
        self.preprocessor = preprocessor

    def _load_model(self):
        import warnings

        import mxnet as mx

        from mxnet import gluon

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
        X = self.preprocessor(X)

        def _transform(X):
            X = mx.nd.array(X)
            db = mx.io.DataBatch(data=(X,))
            self.model.forward(db, is_train=False)
            return self.model.get_outputs()[0].asnumpy()

        if self.memory_demanding:
            features = np.array([_transform(x[None, ...]) for x in X])

            # If we ndim is > than 3. We should stack them all
            # The enroll_features can come from a source where there are `N` samples containing
            # nxd samples
            if features.ndim >= 3:
                features = np.vstack(features)

            return features
        else:
            return _transform(X)

    def __getstate__(self):
        # Handling unpicklable objects

        d = self.__dict__.copy()
        d["model"] = None
        return d

    def _more_tags(self):
        return {"requires_fit": False}


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


from bob.bio.base.algorithm import Distance
from bob.bio.base.pipelines import PipelineSimple
from bob.bio.face.utils import (
    cropped_positions_arcface,
    dnn_default_cropping,
    embedding_transformer,
)


def arcface_template(embedding, annotation_type, fixed_positions=None):
    # DEFINE CROPPING
    cropped_image_size = (112, 112)

    if annotation_type == "eyes-center" or annotation_type == "bounding-box":
        # Hard coding eye positions for backward consistency
        # cropped_positions = {
        cropped_positions = cropped_positions_arcface()
        if annotation_type == "bounding-box":
            # This will allow us to use `BoundingBoxAnnotatorCrop`
            cropped_positions.update(
                {"topleft": (0, 0), "bottomright": cropped_image_size}
            )

    elif isinstance(annotation_type, list):
        cropped_positions = cropped_positions_arcface(annotation_type)
    else:
        cropped_positions = dnn_default_cropping(
            cropped_image_size, annotation_type
        )

    annotator = MTCNN(min_size=40, factor=0.709, thresholds=(0.1, 0.2, 0.2))
    transformer = embedding_transformer(
        cropped_image_size=cropped_image_size,
        embedding=embedding,
        cropped_positions=cropped_positions,
        fixed_positions=fixed_positions,
        color_channel="rgb",
        annotator=annotator,
    )

    algorithm = Distance()

    return PipelineSimple(transformer, algorithm)


def arcface_insightFace_lresnet100(
    annotation_type, fixed_positions=None, memory_demanding=False
):
    return arcface_template(
        ArcFaceInsightFace_LResNet100(memory_demanding=memory_demanding),
        annotation_type,
        fixed_positions,
    )
