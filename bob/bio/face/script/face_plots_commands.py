import click
import numpy as np

from bob.bio.face.reports.arface import arface_report
from bob.bio.face.reports.gbu import gbu_report
from bob.bio.face.reports.ijbc import ijbc_report
from bob.bio.face.reports.mobio import mobio_report
from bob.bio.face.reports.multipie import (
    multipie_expression_report,
    multipie_pose_report,
)
from bob.bio.face.reports.scface import scface_report
from bob.measure.script import common_options


@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.titles_option()
@common_options.eval_option()
@common_options.output_plot_file_option(default_out="multipie.pdf")
@click.pass_context
@click.option(
    "--optimal-threshold",
    is_flag=True,
    help="BE CAREFUL. If this flag is set, it will compute the decision threshold for each sub-protocol.",
)
@click.option(
    "--threshold-eval",
    is_flag=True,
    help="BE CAREFUL. If this flag is set, it will compute the decision threshold using the evaluation set.",
)
@click.option(
    "--fmr-operational-threshold",
    default=1e-3,
    help="FMR operational point used to compute FNMR and FMR on the evaluation set",
)
def multipie_pose(
    ctx,
    scores,
    evaluation,
    output,
    titles,
    optimal_threshold,
    threshold_eval,
    fmr_operational_threshold,
    **kargs,
):
    """plots the multipie POSE report"""

    if len(scores) // 2 != len(titles):
        raise ValueError(
            "Number of scores doesn't match the number of titles. It should be one pair of score files (`dev` and `eval` scores) for one title."
        )

    scores = np.array(scores, dtype="object")

    if evaluation:
        scores_dev = scores[[i for i in list(range(len(scores))) if i % 2 == 0]]
        scores_eval = scores[
            [i for i in list(range(len(scores))) if i % 2 != 0]
        ]
    else:
        scores_dev = scores
        scores_eval = None

    multipie_pose_report(
        scores_dev,
        scores_eval,
        output,
        titles,
        figsize=(8, 4),
        optimal_threshold=optimal_threshold,
        threshold_eval=threshold_eval,
        fmr_threshold=fmr_operational_threshold,
    )

    pass


@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.titles_option()
@common_options.eval_option()
@common_options.output_plot_file_option(default_out="multipie_expression.pdf")
@click.option(
    "--fmr-operational-threshold",
    default=1e-3,
    help="FMR operational point used to compute FNMR and FMR on the evaluation set",
)
@click.pass_context
def multipie_expression(
    ctx, scores, evaluation, output, titles, fmr_operational_threshold, **kargs
):
    """plots the multipie EXPRESSION report"""

    if len(scores) // 2 != len(titles):
        raise ValueError(
            "Number of scores doesn't match the number of titles. It should be one pair of score files (`dev` and `eval` scores) for one title."
        )

    scores = np.array(scores, dtype="object")

    if evaluation:
        scores_dev = scores[[i for i in list(range(len(scores))) if i % 2 == 0]]
        scores_eval = scores[
            [i for i in list(range(len(scores))) if i % 2 != 0]
        ]
    else:
        scores_dev = scores
        scores_eval = None

    multipie_expression_report(
        scores_dev,
        scores_eval,
        output,
        titles,
        figsize=(8, 4),
        fmr_threshold=fmr_operational_threshold,
    )

    pass


@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.titles_option()
@common_options.eval_option()
@common_options.output_plot_file_option(default_out="multipie.pdf")
@click.option(
    "--fmr-operational-threshold",
    default=1e-3,
    help="FMR operational point used to compute FNMR and FMR on the evaluation set",
)
@click.pass_context
def scface_distance(
    ctx, scores, evaluation, output, titles, fmr_operational_threshold, **kargs
):
    """plots the SCFace multi distance"""

    if len(scores) // 2 != len(titles):
        raise ValueError(
            "Number of scores doesn't match the number of titles. It should be one pair of score files (`dev` and `eval` scores) for one title."
        )

    scores = np.array(scores, dtype="object")

    if evaluation:
        scores_dev = scores[[i for i in list(range(len(scores))) if i % 2 == 0]]
        scores_eval = scores[
            [i for i in list(range(len(scores))) if i % 2 != 0]
        ]
    else:
        scores_dev = scores
        scores_eval = None

    scface_report(
        scores_dev,
        scores_eval,
        output,
        titles,
        figsize=(8, 4),
        fmr_threshold=fmr_operational_threshold,
    )

    pass


@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.titles_option()
@common_options.eval_option()
@common_options.output_plot_file_option(default_out="arface.pdf")
@click.option(
    "--fmr-operational-threshold",
    default=1e-3,
    help="FMR operational point used to compute FNMR and FMR on the evaluation set",
)
@click.pass_context
def arface(
    ctx, scores, evaluation, output, titles, fmr_operational_threshold, **kargs
):
    """plots with the arface experiments"""

    if len(scores) // 2 != len(titles):
        raise ValueError(
            "Number of scores doesn't match the number of titles. It should be one pair of score files (`dev` and `eval` scores) for one title."
        )

    scores = np.array(scores, dtype="object")

    if evaluation:
        scores_dev = scores[[i for i in list(range(len(scores))) if i % 2 == 0]]
        scores_eval = scores[
            [i for i in list(range(len(scores))) if i % 2 != 0]
        ]
    else:
        scores_dev = scores
        scores_eval = None

    arface_report(
        scores_dev,
        scores_eval,
        output,
        titles,
        figsize=(8, 4.5),
        fmr_threshold=fmr_operational_threshold,
    )

    pass


@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.titles_option()
@common_options.eval_option()
@common_options.output_plot_file_option(default_out="mobio_gender.pdf")
@click.option(
    "--fmr-operational-threshold",
    default=1e-3,
    help="FMR operational point used to compute FNMR and FMR on the evaluation set",
)
@click.pass_context
def mobio_gender(
    ctx, scores, evaluation, output, titles, fmr_operational_threshold, **kargs
):
    """plots with the arface experiments"""

    if len(scores) // 2 != len(titles):
        raise ValueError(
            "Number of scores doesn't match the number of titles. It should be one pair of score files (`dev` and `eval` scores) for one title."
        )

    scores = np.array(scores, dtype="object")

    if evaluation:
        scores_dev = scores[[i for i in list(range(len(scores))) if i % 2 == 0]]
        scores_eval = scores[
            [i for i in list(range(len(scores))) if i % 2 != 0]
        ]
    else:
        scores_dev = scores
        scores_eval = None

    mobio_report(
        scores_dev,
        scores_eval,
        output,
        titles,
        figsize=(8, 4),
        fmr_threshold=fmr_operational_threshold,
    )

    pass


@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.titles_option()
@common_options.output_plot_file_option(default_out="GBU.pdf")
@click.pass_context
def gbu(ctx, scores, output, titles, **kargs):
    """plots with the GBU experiments"""

    if len(scores) != len(titles):
        raise ValueError("Number of scores doesn't match the number of titles")

    scores = np.array(scores, dtype="object")

    gbu_report(scores, output, titles, figsize=(8, 4))

    pass


@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.titles_option()
@common_options.output_plot_file_option(default_out="ijb-c.pdf")
@click.pass_context
def ijbc(ctx, scores, output, titles, **kargs):
    """plots with the IJB-C experiments"""

    if len(scores) != len(titles):
        raise ValueError("Number of scores doesn't match the number of titles")

    scores = np.array(scores, dtype="object")

    ijbc_report(scores, output, titles, figsize=(8, 6))

    pass
