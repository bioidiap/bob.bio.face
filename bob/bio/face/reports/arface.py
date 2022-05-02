import matplotlib.pyplot as plt
import numpy as np

import bob.measure

from bob.bio.base.score.load import get_split_dataframe


def arface_report(
    scores_dev,
    scores_eval,
    output_filename,
    titles,
    fmr_threshold=1e-3,
    figsize=(16, 8),
    y_abs_max=32,  # Max absolute value for y axis
    colors=plt.cm.tab10.colors,
):

    occlusion_illumination = [
        "illumination",
        "occlusion",
        "both",
    ]

    occlusions = [
        "scarf",
        "sunglasses",
    ]

    # style_iterator = _get_colors_markers()
    # colors = plt.cm.tab20.colors

    eval_fmr_fnmr_occlusion_illumination = dict()
    eval_fmr_fnmr_occlusion = dict()

    for (
        d_scores,
        e_scores,
        title,
    ) in zip(scores_dev, scores_eval, titles):

        eval_fmr_fnmr_occlusion_illumination[title] = []
        eval_fmr_fnmr_occlusion[title] = []

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

        # EVALUATING OCCLUSION AND ILLUMINATION

        # All illumination
        i_eval = impostors_eval.loc[
            (impostors_eval.probe_illumination != "front")
            & (impostors_eval.probe_occlusion == "none")
        ]
        g_eval = genuines_eval.loc[
            (genuines_eval.probe_illumination != "front")
            & (genuines_eval.probe_occlusion == "none")
        ]
        eval_fmr_fnmr_occlusion_illumination[title].append(
            compute_fmr_fnmr(i_eval, g_eval)
        )

        # All occlusions
        i_eval = impostors_eval.loc[
            (impostors_eval.probe_occlusion != "none")
            & (impostors_eval.probe_illumination == "front")
        ]
        g_eval = genuines_eval.loc[
            (genuines_eval.probe_occlusion != "none")
            & (genuines_eval.probe_illumination == "front")
        ]
        eval_fmr_fnmr_occlusion_illumination[title].append(
            compute_fmr_fnmr(i_eval, g_eval)
        )

        # BOTH
        overall_fmr_fnmr = compute_fmr_fnmr(impostors_eval, genuines_eval)
        eval_fmr_fnmr_occlusion_illumination[title].append(overall_fmr_fnmr)

        # EVALUATING DIFFERENT TYPES OF OCCLUSION
        for occlusion in occlusions:
            i_eval = impostors_eval.loc[
                impostors_eval.probe_occlusion == occlusion
            ]
            g_eval = genuines_eval.loc[
                genuines_eval.probe_occlusion == occlusion
            ]

            eval_fmr_fnmr_occlusion[title].append(
                compute_fmr_fnmr(i_eval, g_eval)
            )

    pass

    # Plotting

    #
    # EFFECT OF OCCLUSION TYPES

    # Figure for eval plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    width = 0.8 / len(titles)

    X = np.arange(len(occlusions))

    for i, (title, color) in enumerate(zip(titles, colors)):
        fmrs = [-1 * fmr * 100 for fmr, _ in eval_fmr_fnmr_occlusion[title]]
        fnmrs = [fnmr * 100 for _, fnmr in eval_fmr_fnmr_occlusion[title]]

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

        # Writting the texts on top of the bar plots
        for i, fnmr, fmr in zip(x_axis, fnmrs, fmrs):
            plt.text(
                i - width / 2, fnmr + 0.5, str(round(fnmr, 1)), fontsize=15
            )
            plt.text(
                i - width / 2, fmr - 2.3, str(round(abs(fmr), 1)), fontsize=15
            )

    # Plot finalization
    plt.title(f"FMR vs FNMR.  at Dev. FMR@{fmr_threshold*100}%", fontsize=16)
    ax.set_xlabel("Occlusion types", fontsize=14)
    ax.set_ylabel("FMR(%) vs FNMR(%)                  ", fontsize=18)

    ax.set_xticks(X + 0.5)
    ax.set_xticklabels(occlusions, fontsize=14)

    yticks = np.array(
        [
            -y_abs_max / 4,
            0,
            y_abs_max / 4,
            y_abs_max / 2,
            y_abs_max,
        ]
    )

    ax.set_yticks(yticks)
    ax.set_yticklabels([abs(int(y)) for y in yticks], fontsize=16)

    plt.axhline(0, linestyle="-", color="k")
    plt.ylim([-y_abs_max / 4, y_abs_max + 1])

    plt.legend()
    plt.grid()

    plt.savefig(fig)

    # EFFECT OF ILLUMINATION AND OCCLUSION

    # Figure for eval plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    X = np.arange(len(occlusion_illumination))

    for i, (title, color) in enumerate(zip(titles, colors)):
        fmrs = [
            -1 * fmr * 100
            for fmr, _ in eval_fmr_fnmr_occlusion_illumination[title]
        ]
        fnmrs = [
            fnmr * 100
            for _, fnmr in eval_fmr_fnmr_occlusion_illumination[title]
        ]

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
        # Writting the texts on top of the bar plots
        for i, fnmr, fmr in zip(x_axis, fnmrs, fmrs):
            plt.text(
                i - width / 2,
                fnmr + 0.4,
                str(int(fnmr)) if fnmr == 0 else str(round(fnmr, 1)),
                fontsize=15,
            )
            plt.text(
                i - width / 2,
                fmr - 0.9,
                str(int(fmr)),
                fontsize=15,
            )

    plt.title(f"FMR vs FNMR.  at Dev. FMR@{fmr_threshold*100}%", fontsize=16)
    ax.set_xlabel("Occlusion and illumination", fontsize=14)
    ax.set_ylabel("FMR(%) vs FNMR(%)            ", fontsize=18)

    ax.set_xticks(X + 0.5)
    ax.set_xticklabels(occlusion_illumination, fontsize=14)

    yticks = np.array([-y_abs_max / 4, 0, y_abs_max / 4, y_abs_max / 2])
    ax.set_yticks(yticks)
    ax.set_yticklabels([abs(int(y)) for y in yticks], fontsize=16)

    plt.axhline(0, linestyle="-", color="k")
    plt.ylim([-y_abs_max / 4, y_abs_max / 2])

    plt.legend()
    plt.grid()
    plt.savefig(fig)
