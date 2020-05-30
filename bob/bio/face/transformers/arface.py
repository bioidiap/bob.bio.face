#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import os
from sklearn.base import TransformerMixin, BaseEstimator
from .tensorflow_compat_v1 import TensorflowCompatV1
from bob.io.image import to_matplotlib
import numpy as np


class ArcFace_InsightFaceTF(TensorflowCompatV1):
    """
    Models copied from
    https://github.com/luckycallor/InsightFace-tensorflow/blob/master/backbones/utils.py

    The input shape of this model is :math:`3 \times 112 \times 112`
    The output embedding is :math:`n \times 512`, where :math:`n` is the number of samples

    """

    def __init__(self):

        bob_rc_variable = "bob.bio.face.arcface_tf_path"
        urls = [
            "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/arcface_insight_tf.tar.gz"
        ]
        model_subdirectory = "arcface_tf_path"

        checkpoint_filename = self.get_modelpath(bob_rc_variable, model_subdirectory)
        self.download_model(checkpoint_filename, urls)

        input_shape = (1, 112, 112, 3)
        architecture_fn = init_network

        super().__init__(checkpoint_filename, input_shape, architecture_fn)

    def transform(self, data):

        # https://github.com/luckycallor/InsightFace-tensorflow/blob/master/evaluate.py#L42
        data = np.asarray(data)
        data = data / 127.5 - 1.0

        return super().transform(data)

    def load_model(self):

        self.input_tensor = tf.compat.v1.placeholder(
            dtype=tf.float32,
            shape=self.input_shape,
            name="input_image",
        )

        prelogits = self.architecture_fn(self.input_tensor)
        self.embedding = prelogits


        # Initializing the variables of the current graph
        self.session = tf.compat.v1.Session()
        self.session.run(tf.compat.v1.global_variables_initializer())

        # Loading the last checkpoint and overwriting the current variables
        saver = tf.compat.v1.train.Saver()

        if os.path.splitext(self.checkpoint_filename)[1] == ".meta":
            saver.restore(
                self.session,
                tf.train.latest_checkpoint(os.path.dirname(self.checkpoint_filename)),
            )
        elif os.path.isdir(self.checkpoint_filename):
            saver.restore(self.session, tf.train.latest_checkpoint(self.checkpoint_filename))
        else:
            saver.restore(self.session, self.checkpoint_filename)

        self.loaded = True


###########################
# CODE COPIED FROM
# https://github.com/luckycallor/InsightFace-tensorflow/blob/master/backbones/utils.py
###########################

import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import namedtuple


def init_network(input_tensor):

    with tf.variable_scope("embd_extractor", reuse=False):
        arg_sc = resnet_arg_scope()
        with slim.arg_scope(arg_sc):
            net, _ = resnet_v2_m_50(input_tensor, is_training=False, return_raw=True)

            net = slim.batch_norm(net, activation_fn=None, is_training=False)
            net = slim.dropout(net, keep_prob=1, is_training=False)
            net = slim.flatten(net)
            net = slim.fully_connected(net, 512, normalizer_fn=None, activation_fn=None)
            net = slim.batch_norm(
                net, scale=False, activation_fn=None, is_training=False
            )
            # end_points['embds'] = net

    return net


def resnet_v2_m_50(
    inputs,
    num_classes=None,
    is_training=True,
    return_raw=True,
    global_pool=True,
    output_stride=None,
    spatial_squeeze=True,
    reuse=None,
    scope="resnet_v2_50",
):
    """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_v2_block("block1", base_depth=16, num_units=3, stride=2),
        resnet_v2_block("block2", base_depth=32, num_units=4, stride=2),
        resnet_v2_block("block3", base_depth=64, num_units=14, stride=2),
        resnet_v2_block("block4", base_depth=128, num_units=3, stride=2),
    ]
    return resnet_v2_m(
        inputs,
        blocks,
        num_classes,
        is_training=is_training,
        return_raw=return_raw,
        global_pool=global_pool,
        output_stride=output_stride,
        include_root_block=True,
        spatial_squeeze=spatial_squeeze,
        reuse=reuse,
        scope=scope,
    )


def resnet_v2_block(scope, base_depth, num_units, stride):
    return Block(
        scope,
        block,
        [{"depth": base_depth * 4, "stride": stride}]
        + (num_units - 1) * [{"depth": base_depth * 4, "stride": 1}],
    )


class Block(namedtuple("Block", ["scope", "unit_fn", "args"])):
    """A named tuple describing a ResNet block.
    Its parts are:
        scope: The scope of the `Block`.
        unit_fn: The ResNet unit function which takes as input a `Tensor` and returns another `Tensor` with the output of the ResNet unit.
        args: A list of length equal to the number of units in the `Block`. The list contains one (depth, depth_bottleneck, stride) tuple for each unit in the block to serve as argument to unit_fn.
    """

    pass


def resnet_v2_m(
    inputs,
    blocks,
    num_classes=None,
    is_training=True,
    return_raw=True,
    global_pool=True,
    output_stride=None,
    include_root_block=True,
    spatial_squeeze=True,
    reuse=None,
    scope=None,
):
    with tf.variable_scope(scope, "resnet_v2", [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + "_end_points"
        with slim.arg_scope(
            [slim.conv2d, bottleneck, stack_blocks_dense],
            outputs_collections=end_points_collection,
        ):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError(
                                "The output_stride needs to be a multiple of 4."
                            )
                        output_stride /= 4
                    with slim.arg_scope(
                        [slim.conv2d], activation_fn=None, normalizer_fn=None
                    ):
                        net = conv2d_same(net, 64, 3, stride=1, scope="conv1")
                    # net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                net = stack_blocks_dense(net, blocks, output_stride)
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection
                )
                if return_raw:
                    return net, end_points
                net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope="postnorm")
                end_points[sc.name + "/postnorm"] = net

                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], name="pool5", keep_dims=True)
                    end_points["global_pool"] = net

                if num_classes:
                    net = slim.conv2d(
                        net,
                        num_classes,
                        [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        scope="logits",
                    )
                    end_points[sc.name + "/logits"] = net
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")
                        end_points[sc.name + "/spatial_squeeze"] = net
                    end_points["predictions"] = slim.softmax(net, scope="predictions")
                return net, end_points


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    if stride == 1:
        return slim.conv2d(
            inputs,
            num_outputs,
            kernel_size,
            stride=1,
            rate=rate,
            padding="SAME",
            scope=scope,
        )
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(
            inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
        )  # zero padding
        return slim.conv2d(
            inputs,
            num_outputs,
            kernel_size,
            stride=stride,
            rate=rate,
            padding="VALID",
            scope=scope,
        )


@slim.add_arg_scope
def stack_blocks_dense(
    net,
    blocks,
    output_stride=None,
    store_non_strided_activations=False,
    outputs_collections=None,
):
    current_stride = 1
    rate = 1

    for block in blocks:
        with tf.variable_scope(block.scope, "block", [net]) as sc:
            block_stride = 1
            for i, unit in enumerate(block.args):
                if store_non_strided_activations and i == len(block.args) - 1:
                    block_stride = unit.get("stride", 1)
                    unit = dict(unit, stride=1)
                with tf.variable_scope("unit_%d" % (i + 1), values=[net]):
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
                        rate *= unit.get("stride", 1)
                    else:
                        net = block.unit_fn(net, rate=1, **unit)
                        current_stride *= unit.get("stride", 1)
                        if output_stride is not None and current_stride > output_stride:
                            raise ValueError(
                                "The target output_stride cannot be reached."
                            )
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

            if output_stride is not None and current_stride == output_stride:
                rate *= block_stride
            else:
                net = subsample(net, block_stride)
                current_stride *= block_stride
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError("The target output_stride cannot be reached.")
    if output_stride is not None and current_stride != output_stride:
        raise ValueError("The target output_stride cannot be reached.")
    return net


def block(inputs, depth, stride, rate=1, outputs_collections=None, scope=None):
    with tf.variable_scope(scope, "block_v2", [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.leaky_relu, scope="preact")
        if depth == depth_in:
            shortcut = subsample(inputs, stride, "shortcut")
        else:
            shortcut = slim.conv2d(
                preact,
                depth,
                [1, 1],
                stride=stride,
                normalizer_fn=None,
                activation_fn=None,
                scope="shortcut",
            )

        residual = conv2d_same(preact, depth, 3, stride, rate=rate, scope="conv1")
        residual = slim.conv2d(
            residual,
            depth,
            [3, 3],
            stride=1,
            normalizer_fn=None,
            activation_fn=None,
            scope="conv2",
        )
        # residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


@slim.add_arg_scope
def bottleneck(
    inputs,
    depth,
    depth_bottleneck,
    stride,
    rate=1,
    outputs_collections=None,
    scope=None,
):
    with tf.variable_scope(scope, "bottleneck_v2", [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.leaky_relu, scope="preact")
        if depth == depth_in:
            shortcut = subsample(inputs, stride, "shortcut")
        else:
            shortcut = slim.conv2d(
                preact,
                depth,
                [1, 1],
                stride=stride,
                normalizer_fn=None,
                activation_fn=None,
                scope="shortcut",
            )

        residual = slim.conv2d(
            preact, depth_bottleneck, [1, 1], stride=1, scope="conv1"
        )
        residual = conv2d_same(
            residual, depth_bottleneck, 3, stride, rate=rate, scope="conv2"
        )
        residual = slim.conv2d(
            residual,
            depth,
            [1, 1],
            stride=1,
            normalizer_fn=None,
            activation_fn=None,
            scope="conv3",
        )

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(
            inputs, [1, 1], stride=factor, scope=scope
        )  # padding='VALID'


def resnet_arg_scope(
    weight_decay=0.0001,
    batch_norm_decay=0.9,
    batch_norm_epsilon=2e-5,
    batch_norm_scale=True,
    activation_fn=tf.nn.leaky_relu,
    use_batch_norm=True,
    batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,
):
    batch_norm_params = {
        "decay": batch_norm_decay,
        "epsilon": batch_norm_epsilon,
        "scale": batch_norm_scale,
        "updates_collections": batch_norm_updates_collections,
        "fused": None,  # Use fused batch norm if possible.
        "param_regularizers": {"gamma": slim.l2_regularizer(weight_decay)},
    }

    with slim.arg_scope(
        [slim.conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
        activation_fn=activation_fn,
        normalizer_fn=slim.batch_norm if use_batch_norm else None,
        normalizer_params=batch_norm_params,
    ):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding="SAME") as arg_sc:
                return arg_sc
