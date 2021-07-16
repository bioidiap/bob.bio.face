import sys
import time

import dask
from bob.extension import rc
from dask.distributed import Client
from dask_jobqueue import SGECluster

verbose = 3

CONFIG = {
    # distributed training
    "n-workers": 1,
    # batch size originally was 128
    "batch-size": 128,
    # number train samples to consider before saving one epoch, originally it was 256K
    "n-train-samples-per-epoch": 256_000 * 1,
    "shuffle-buffer": int(1e6),
    ## Backbone
    "backbone": "iresnet100",
    "use-l2-regularizer": False,
    ## head
    "s": 30,
    "bottleneck": 512,
    "m": 0.5,
    # Training parameters
    "dropout-rate": 0.0,
    "train-tf-record-path": f"{rc['face-tfrecords']}/112x112_2/webface_260M/*.tfrecord",
    "face-size": 112,
    "n-classes": 617970,
    "validation-tf-record-path": f"{rc['face-tfrecords']}/112x112_2/lfw_sharded/*.tfrecords",
    "checkpoint-path": f"{rc['temp']}/hardening/arcface_sgd_prelu/i100_w8_b128_fp16_drp0_slow_decay_webface",
    "epochs": 6000,
}

# strategy_fn = "mirrored-strategy"
mixed_precision_policy = "mixed_float16"


def setup_dask(n_workers):

    dask.config.set({"jobqueue.sge.walltime": None})
    dask.config.set(
        {"distributed.worker.memory.target": False}
    )  # Avoid spilling to disk
    dask.config.set(
        {"distributed.worker.memory.spill": False}
    )  # Avoid spilling to disk

    cluster = SGECluster(
        queue="q_gpu",
        # # 164m is 2h44m
        # extra=["--lifetime", "164m", "--lifetime-stagger", "4m"],
        memory="600GiB",  # this is for dask not SGE
        cores=1,
        processes=1,
        log_directory="./logs",
        # silence_logs="info",
        local_directory=None,
        resource_spec="q_gpu=TRUE,hostname=vgni*",
        project=rc.get("sge.project"),
        env_extra=[
            "export PYTHONUNBUFFERED=1",
            f"export PYTHONPATH={':'.join(sys.path)}",
            "ulimit -a",  # to see if no virtual max memory is set
            # # Need to tell dask workers not to use daemonic processes
            # # see: https://github.com/dask/distributed/issues/2718
            # "export DASK_DISTRIBUTED__WORKER__DAEMON=False",
        ],
    )
    dask_client = Client(cluster, timeout="2m")
    cluster.scale(n_workers)
    print(f"Waiting for {n_workers} dask workers to be allocated ...")
    dask_client.wait_for_workers(n_workers=n_workers, timeout="2m")
    print(f"All requested {n_workers} dask workers are ready!")

    for _ in range(30):
        if len(cluster.requested) == n_workers:
            break
        time.sleep(1)
    if not len(cluster.requested) == n_workers:
        raise ValueError(
            f"cluster.requested is {cluster.requested} but n_workers is {n_workers}"
        )

    return dask_client


dask_client = setup_dask(CONFIG["n-workers"])
