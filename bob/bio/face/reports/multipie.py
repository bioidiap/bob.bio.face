import itertools

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages

import bob.measure

from bob.bio.base.score.load import get_split_dataframe


def _get_colors_markers():
    l_styles = ["-", "--", "-.", ":", "-.-"]
    m_styles = [".", "o", "^", "*", "o"]

    return itertools.product(l_styles, m_styles)


def multipie_pose_report(
    scores_dev,
    scores_eval,
    output_filename,
    titles,
    figsize=(16, 8),
    fmr_threshold=1e-3,
    colors=plt.cm.tab10.colors,
    optimal_threshold=False,
    threshold_eval=False,
):
    """
    Compute the multipie pose report, ploting the FNMR for each view point

    Parameters
    ----------

    scores_dev:

    scores_eval:

    output_filename:

    titles:

    figsize=(16, 8):

    fmr_threshold:

    colors:
       Color palete

    optimal_threshold: bool
      If set it to `True`, it will compute one decision threshold for each
      subprotocol (for each pose). Default to false.

    threshold_eval: bool
      If set it to `True` it will compute the threshold using the evaluation set.
      Default obviouslly to `False`.

    """

    cameras = [
        "11_0",
        "12_0",
        "09_0",
        "08_0",
        "13_0",
        "14_0",
        "05_1",
        "05_0",
        "04_1",
        "19_0",
        "20_0",
        "01_0",
        "24_0",
    ]

    angles = np.linspace(-90, 90, len(cameras))
    camera_to_angle = dict(zip(cameras, angles))

    pdf = PdfPages(output_filename)

    # Figure for eval plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    style_iterator = _get_colors_markers()

    for d_scores, e_scores, title, (linestyle, marker), color in zip(
        scores_dev, scores_eval, titles, style_iterator, colors
    ):

        # Load the score files and fill in the angle associated to each camera
        impostors_dev, genuines_dev = get_split_dataframe(d_scores)
        impostors_eval, genuines_eval = get_split_dataframe(e_scores)

        # loading the dask dataframes
        impostors_dev = impostors_dev.compute()
        genuines_dev = genuines_dev.compute()
        impostors_eval = impostors_eval.compute()
        genuines_eval = genuines_eval.compute()

        # Appending the angle
        impostors_dev["angle"] = impostors_dev["probe_camera"].map(
            camera_to_angle
        )
        genuines_dev["angle"] = genuines_dev["probe_camera"].map(
            camera_to_angle
        )
        impostors_eval["angle"] = impostors_eval["probe_camera"].map(
            camera_to_angle
        )
        genuines_eval["angle"] = genuines_eval["probe_camera"].map(
            camera_to_angle
        )

        angles = []
        eval_fmr_fnmr = []

        # Compute the min. HTER threshold on the Dev set
        threshold = None
        if not optimal_threshold:

            if threshold_eval:
                threshold = bob.measure.far_threshold(
                    impostors_eval["score"].to_numpy(),
                    genuines_eval["score"].to_numpy(),
                    fmr_threshold,
                )
            else:
                threshold = bob.measure.far_threshold(
                    impostors_dev["score"].to_numpy(),
                    genuines_dev["score"].to_numpy(),
                    fmr_threshold,
                )

        # Run the analysis per view angle
        for ((angle, i_dev), (_, g_dev), (_, i_eval), (_, g_eval),) in zip(
            impostors_dev.groupby("angle"),
            genuines_dev.groupby("angle"),
            impostors_eval.groupby("angle"),
            genuines_eval.groupby("angle"),
        ):

            if optimal_threshold:
                if threshold_eval:
                    threshold = bob.measure.far_threshold(
                        i_eval["score"].to_numpy(),
                        g_eval["score"].to_numpy(),
                        fmr_threshold,
                    )
                else:
                    threshold = bob.measure.far_threshold(
                        i_dev["score"].to_numpy(),
                        g_dev["score"].to_numpy(),
                        fmr_threshold,
                    )

            eval_fmr, eval_fnmr = bob.measure.farfrr(
                i_eval["score"].to_numpy(),
                g_eval["score"].to_numpy(),
                threshold,
            )
            # eval_fnmr = (eval_fnmr + eval_fmr) / 2

            angles.append(angle)
            eval_fmr_fnmr.append([eval_fmr, eval_fnmr])

            # eval_hter = (1 / 2 * (eval_far + eval_frr)) * 100
            # eval_hter = eval_frr * 100
            # eval_hters.append(eval_hter)

        # Update plots

        fnrms = [fnmr * 100 for fmr, fnmr in eval_fmr_fnmr]
        # fmrs = [-fmr * 100 for fmr, fnmr in eval_fmr_fnmr]
        plt.plot(
            angles,
            fnrms,
            label=title,
            linestyle=linestyle,
            marker=marker,
            linewidth=2,
            color=color,
        )

        # plt.plot(
        #    angles, fnrms, label=title, linestyle=linestyle, marker=marker, linewidth=2,
        # )
    if threshold_eval:
        plt.title(f"FNMR. at Eval. FMR@{fmr_threshold*100}%", fontsize=16)
    else:
        plt.title(f"FNMR. at Dev. FMR@{fmr_threshold*100}%", fontsize=16)
    plt.xlabel("Angle", fontsize=12)
    plt.ylabel("FNMR(%)", fontsize=14)
    plt.legend()
    plt.grid()

    # Plot finalization

    xticks = [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]
    plt.xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=10)

    yticks = np.array([0, 20, 40, 60, 80, 100])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=12)

    pdf.savefig(fig)
    pdf.close()


def multipie_expression_report(
    scores_dev,
    scores_eval,
    output_filename,
    titles,
    fmr_threshold=1e-3,
    figsize=(16, 8),
    y_abs_max=60,  # Max absolute value for y axis
    colors=plt.cm.tab10.colors,
):

    # "all",
    expressions = [
        "neutral",
        "smile",
        "surprise",
        "squint",
        "disgust",
        "scream",
    ]

    # style_iterator = _get_colors_markers()
    # colors = plt.cm.tab20.colors

    eval_fmr_fnmr_expressions = dict()

    for (
        d_scores,
        e_scores,
        title,
    ) in zip(scores_dev, scores_eval, titles):

        eval_fmr_fnmr_expressions[title] = []

        # Load the score files and fill in the angle associated to each camera
        impostors_dev, genuines_dev = get_split_dataframe(d_scores)
        impostors_eval, genuines_eval = get_split_dataframe(e_scores)

        # loading the dask dataframes
        impostors_dev = impostors_dev.compute()
        genuines_dev = genuines_dev.compute()
        impostors_eval = impostors_eval.compute()
        genuines_eval = genuines_eval.compute()

        # Computing the threshold combining all distances
        threshold = bob.measure.far_threshold(
            impostors_dev["score"].to_numpy(),
            genuines_dev["score"].to_numpy(),
            fmr_threshold,
        )

        def compute_fmr_fnmr(impostor_scores, genuine_scores):
            eval_fmr, eval_fnmr = bob.measure.farfrr(
                impostor_scores["score"].to_numpy(),
                genuine_scores["score"].to_numpy(),
                threshold,
            )
            return eval_fmr, eval_fnmr

        # All expressions
        # i_eval = impostors_eval
        # g_eval = genuines_eval
        # eval_fmr_fnmr_expressions[title].append(compute_fmr_fnmr(i_eval, g_eval))

        # EVALUATING DIFFERENT TYPES OF EXPRESSION
        for expression in expressions:
            i_eval = impostors_eval.loc[
                impostors_eval.probe_expression == expression
            ]
            g_eval = genuines_eval.loc[
                genuines_eval.probe_expression == expression
            ]

            eval_fmr_fnmr_expressions[title].append(
                compute_fmr_fnmr(i_eval, g_eval)
            )

    pass

    # Plotting
    pdf = PdfPages(output_filename)

    # Figure for eval plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    width = 0.8 / len(titles)

    X = np.arange(len(expressions))

    for i, (title, color) in enumerate(zip(titles, colors)):
        fmrs = [-1 * fmr * 100 for fmr, _ in eval_fmr_fnmr_expressions[title]]
        fnmrs = [fnmr * 100 for _, fnmr in eval_fmr_fnmr_expressions[title]]

        x_axis = X + (i + 1) * width - width / 2
        ax.bar(
            x_axis,
            fmrs,
            width,
            label=title,
            color=color,
            alpha=1,
            hatch="\\",
        )
        ax.bar(
            x_axis,
            fnmrs,
            width,
            color=color,
            alpha=0.5,
            hatch="/",
        )
        # str(int(fnmr)) if fnmr == 0 else str(round(fnmr, 1))
        # Writting the texts on top of the bar plots
        for i, fnmr, fmr in zip(x_axis, fnmrs, fmrs):
            plt.text(
                i - width / 2,
                fnmr + 0.8,
                str(int(fnmr)),
                fontsize=10,
            )
            plt.text(
                i - width / 2,
                fmr - 2.8,
                str(int(abs(fmr))),
                fontsize=10,
            )

    # Plot finalization
    plt.title(f"FMR vs FNMR.  at Dev. FMR@{fmr_threshold*100}%", fontsize=16)
    ax.set_xlabel("Expressions", fontsize=12)
    ax.set_ylabel("FMR(%) vs FNMR(%)         ", fontsize=14)

    ax.set_xticks(X + 0.5)
    ax.set_xticklabels(expressions)

    yticks = np.array(
        [
            -y_abs_max / 8,
            0,
            y_abs_max / 4,
            y_abs_max / 2,
            y_abs_max,
        ]
    )

    ax.set_yticks(yticks)
    ax.set_yticklabels([int(abs(y)) for y in yticks], fontsize=16)

    plt.axhline(0, linestyle="-", color="k")
    plt.ylim([-y_abs_max / 8, y_abs_max + 1])

    plt.legend()
    plt.grid()

    pdf.savefig(fig)

    pdf.close()
