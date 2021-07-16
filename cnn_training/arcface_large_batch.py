"""Train face recognition baselines using ARCFace loss
To run copy and modify arcface_large_batch_sample_config.py in a new file (e.g. my_config.py) and run::

    bob keras fit -vvv my_config.py src/bob.bio.face/cnn_training/arcface_large_batch.py
"""

import tensorflow as tf
import tensorflow_addons as tfa
from bob.bio.face.tensorflow.preprocessing import prepare_dataset
from bob.extension import rc
from bob.learn.tensorflow.callbacks import add_backup_callback
from bob.learn.tensorflow.layers import add_bottleneck
from bob.learn.tensorflow.metrics import EmbeddingAccuracy
from bob.learn.tensorflow.models import ArcFaceLayer
from bob.learn.tensorflow.models.iresnet import iresnet50
from bob.learn.tensorflow.models.iresnet import iresnet100

# CONFIG comes from another python file
CONFIG = locals()["CONFIG"]

# accepted CONFIG parameters
BACKBONE = CONFIG.pop("backbone")
BATCH_SIZE = CONFIG.pop("batch-size")
BOTTLENECK = CONFIG.pop("bottleneck")
CHECKPOINT_PATH = CONFIG.pop("checkpoint-path")
DROPOUT_RATE = CONFIG.pop("dropout-rate")
EPOCHS = CONFIG.pop("epochs")
FACE_SIZE = CONFIG.pop("face-size")
M = CONFIG.pop("m")
N_CLASSES = CONFIG.pop("n-classes")
N_TRAIN_SAMPLES_PER_EPOCH = CONFIG.pop("n-train-samples-per-epoch")
N_WORKERS = CONFIG.pop("n-workers")
S = CONFIG.pop("s")
SHUFFLE_BUFFER = CONFIG.pop("shuffle-buffer")
TRAIN_TF_RECORD_PATH = CONFIG.pop("train-tf-record-path")
USE_L2_REGULARIZER = CONFIG.pop("use-l2-regularizer")
VALIDATION_TF_RECORD_PATH = CONFIG.pop("validation-tf-record-path")

if CONFIG:
    raise ValueError(f"Got unknown CONFIG values: {CONFIG}")

initial_lr = 0.1 / 512 * BATCH_SIZE * N_WORKERS


params = {
    "optimizer": {
        "type": "sgdw",
        "sgdw": {
            # multiply momentum by initial_lr to roughly match pytorch formula in sgd
            "momentum": min(0.9 * initial_lr, 0.999),
            "nesterov": False,
            "weight_decay": 5e-4,
            # normally you don't apply weight decay on batch norm layers and biases
            # but the sgdw optimizer doesn't support this
            # only works on Adam implementation from tensorflow/models/official
            # "exclude_from_weight_decay": ["bn/", "/bias"],
        },
    },
    "learning_rate": {
        "type": "constant",
        "constant": {
            "learning_rate": initial_lr,
        },
    },
}


def optimizer_fn():
    if (
        params["optimizer"]["type"] == "sgdw"
        and params["learning_rate"]["type"] == "constant"
    ):
        kwargs = dict(params["optimizer"]["sgdw"])
        kwargs["learning_rate"] = params["learning_rate"]["constant"]["learning_rate"]
        optimizer = tfa.optimizers.SGDW(**kwargs)
    else:
        # needs https://github.com/tensorflow/models/tree/master/official installed
        from official.modeling.optimization import OptimizationConfig
        from official.modeling.optimization import OptimizerFactory

        opt_config = OptimizationConfig(params)
        opt_factory = OptimizerFactory(opt_config)
        lr = opt_factory.build_learning_rate()
        optimizer = opt_factory.build_optimizer(lr)
    return optimizer


################################
# DATA SPECIFICATION
###############################
DATA_SHAPES = dict()

# Inputs with 182x182 are cropped to 160x160
DATA_SHAPES[182] = 160
DATA_SHAPES[112] = 112
DATA_SHAPES[126] = 112


# SHAPES EXPECTED FROM THE DATASET USING THIS BACKBONE
DATA_TYPE = tf.uint8
AUTOTUNE = tf.data.experimental.AUTOTUNE

# HERE WE VALIDATE WITH LFW RUNNING A
# INFORMATION ABOUT THE VALIDATION SET
# there are 3200 samples in the validation set
VALIDATION_SAMPLES = 3200
VALIDATION_BATCH_SIZE = 38 * N_WORKERS


##########################
# bob keras fit parameters
##########################
# epochs given to model.fit(...)
epochs = EPOCHS

# number of training steps to do before validating a model. This also defines an epoch
# for keras which is not really true. We want to evaluate every 180000 (90 * 2000)
# samples
# KERAS_EPOCH_MULTIPLIER = 6
# 256000 // 128 == 2000
steps_per_epoch = N_TRAIN_SAMPLES_PER_EPOCH // (
    BATCH_SIZE * N_WORKERS
)

validation_steps = VALIDATION_SAMPLES // VALIDATION_BATCH_SIZE


def create_model(
    n_classes, model_spec, backbone, bottleneck, dropout_rate, input_shape
):
    arch = {
        "iresnet50": iresnet50,
        "iresnet100": iresnet100,
    }[BACKBONE]
    pre_model = arch(
        input_shape=input_shape,
    )

    # Adding the bottleneck
    pre_model = add_bottleneck(
        pre_model,
        bottleneck_size=bottleneck,
        dropout_rate=dropout_rate,
        w_decay=2.5e-4 if USE_L2_REGULARIZER else None,
        batch_norm_decay=0.9,
        batch_norm_epsilon=1e-5,
        batch_norm_scale=False,
        use_bias=False,
    )

    embeddings = pre_model.get_layer("embeddings").output
    labels = tf.keras.layers.Input([], name="label")

    # Creating the specific models
    if "arcface" in model_spec:
        logits_arcface = ArcFaceLayer(
            n_classes,
            s=model_spec["arcface"]["s"],
            m=model_spec["arcface"]["m"],
            arc=True,
            dtype="float32",
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01),
        )(embeddings, labels)
        arc_model = tf.keras.Model(
            inputs=(pre_model.input, labels), outputs=[logits_arcface, embeddings]
        )
    else:
        raise ValueError(f"Unknown model_spec: {model_spec}")

    return arc_model


def build_and_compile_models(
    n_classes,
    optimizer_fn,
    model_spec,
    backbone,
    bottleneck,
    dropout_rate,
    input_shape,
):
    arc_model = create_model(
        n_classes,
        model_spec,
        backbone,
        bottleneck,
        dropout_rate,
        input_shape,
    )

    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, name="cross_entropy"
    )

    # Compile the Cross-entropy model if the case
    loss = [cross_entropy, None]
    metrics = {"arc_face_logits": "accuracy", "embeddings": EmbeddingAccuracy()}

    arc_model.compile(
        optimizer=optimizer_fn(),
        loss=loss,
        metrics=metrics,
        # run_eagerly=True,
    )

    return arc_model


def output_shape():
    face_size = int(FACE_SIZE)
    OUTPUT_SHAPE = (DATA_SHAPES[face_size], DATA_SHAPES[face_size])
    return OUTPUT_SHAPE


def model_fn():
    model_spec = dict()
    model_spec["arcface"] = dict()
    model_spec["arcface"]["m"] = float(M)
    model_spec["arcface"]["s"] = int(S)

    if optimizer_fn is None:
        raise ValueError("optimizer_fn not found")

    arc_model = build_and_compile_models(
        n_classes=int(N_CLASSES),
        optimizer_fn=optimizer_fn,
        model_spec=model_spec,
        backbone=BATCH_SIZE,
        bottleneck=int(BOTTLENECK),
        dropout_rate=float(DROPOUT_RATE),
        input_shape=output_shape() + (3,),
    )

    return arc_model


def data_shape():
    face_size = int(FACE_SIZE)
    DATA_SHAPE = (face_size, face_size, 3)
    return DATA_SHAPE


def label_as_input(x, y):
    return ((x, y), y)


def train_input_fn(ctx=None):
    train_ds = prepare_dataset(
        tf_record_paths=TRAIN_TF_RECORD_PATH,
        batch_size=int(BATCH_SIZE) * int(N_WORKERS),
        # repeat indefinitely for https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#evaluation
        epochs=-1,
        data_shape=data_shape(),
        output_shape=output_shape(),
        shuffle=True,
        augment=True,
        shuffle_buffer=SHUFFLE_BUFFER,
        ctx=ctx,
    )
    train_ds = train_ds.map(label_as_input)
    return train_ds


def eval_input_fn(ctx=None):
    val_ds = prepare_dataset(
        tf_record_paths=VALIDATION_TF_RECORD_PATH,
        batch_size=VALIDATION_BATCH_SIZE,
        epochs=-1,
        data_shape=data_shape(),
        output_shape=output_shape(),
        shuffle=False,
        augment=False,
        shuffle_buffer=SHUFFLE_BUFFER,
        ctx=ctx,
    )
    val_ds = val_ds.map(label_as_input)
    return val_ds


callbacks = {
    "reduce-lr-on-plateau": tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss",
        factor=0.9,
        patience=10,
        verbose=1,
        mode="min",
        min_delta=1,
        cooldown=0,
        min_lr=1e-4,
    ),
    "epochs": tf.keras.callbacks.ModelCheckpoint(
        f"{CHECKPOINT_PATH}/epochs/epoch.{{epoch:04d}}",
        verbose=1,
        # save less frequently when increasing batch-size and n-workers
        save_freq=(steps_per_epoch * BATCH_SIZE * N_WORKERS) // 100,
    ),
    "best": tf.keras.callbacks.ModelCheckpoint(
        f"{CHECKPOINT_PATH}/best",
        verbose=1,
        monitor="val_embeddings_embedding_accuracy",
        mode="max",
        save_best_only=True,
        save_freq="epoch",
    ),
    "nan": tf.keras.callbacks.TerminateOnNaN(),
    "tensorboard": tf.keras.callbacks.TensorBoard(
        log_dir=f"{CHECKPOINT_PATH}/logs",
        update_freq=15,
        profile_batch=0,
        write_steps_per_second=True,
    ),
}

callbacks = add_backup_callback(
    callbacks, backup_dir=f"{CHECKPOINT_PATH}/backup"
)
