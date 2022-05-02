import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages

import bob.measure

from bob.bio.base.score.load import get_split_dataframe


def scface_report(
    scores_dev,
    scores_eval,
    output_filename,
    titles,
    fmr_threshold=1e-3,
    figsize=(16, 8),
    colors=plt.cm.tab10.colors,
):

    distances = [
        "close",
        "medium",
        "far",
    ]
    # "combined",

    eval_fmr_fnmr = dict()

    for (
        d_scores,
        e_scores,
        title,
    ) in zip(scores_dev, scores_eval, titles):

        eval_fmr_fnmr[title] = []

        # Load the score files and fill in the angle associated to each camera
        impostors_dev, genuines_dev = get_split_dataframe(d_scores)
        impostors_eval, genuines_eval = get_split_dataframe(e_scores)

        # loading the dask dataframes
        impostors_dev = impostors_dev.compute()
        genuines_dev = genuines_dev.compute()
        impostors_eval = impostors_eval.compute()
        genuines_eval = genuines_eval.compute()
        # Computing the threshold combining all distances

        # Computing the decision threshold
        i_dev = impostors_dev["score"].to_numpy()
        g_dev = genuines_dev["score"].to_numpy()

        threshold = bob.measure.far_threshold(
            i_dev,
            g_dev,
            far_value=fmr_threshold,
        )

        def compute_fmr_fnmr(impostor_scores, genuine_scores):
            eval_fmr, eval_fnmr = bob.measure.farfrr(
                impostor_scores["score"].to_numpy(),
                genuine_scores["score"].to_numpy(),
                threshold,
            )
            return eval_fmr, eval_fnmr

        for distance in distances:

            i_eval = impostors_eval.loc[
                impostors_eval.probe_distance == distance
            ]
            g_eval = genuines_eval.loc[genuines_eval.probe_distance == distance]

            eval_fmr_fnmr[title].append(compute_fmr_fnmr(i_eval, g_eval))

        # Combined
        # eval_fmr_fnmr[title].append(compute_fmr_fnmr(impostors_eval, genuines_eval))

    pass

    # Plotting
    pdf = PdfPages(output_filename)

    # Figure for eval plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    width = 0.8 / len(titles)
    X = np.arange(len(distances))

    for i, (title, color) in enumerate(zip(titles, colors)):
        fmrs = [-1 * fmr * 100 for fmr, _ in eval_fmr_fnmr[title]]
        fnmrs = [fnmr * 100 for _, fnmr in eval_fmr_fnmr[title]]
        x_axis = X + (i + 1) * width - width / 2
        ax.bar(
            x_axis,
            fmrs,
            width,
            label=title,
            color=color,
            alpha=0.75,
            hatch="\\",
        )
        ax.bar(x_axis, fnmrs, width, color=color, alpha=0.5, hatch="/")

        # Writting the texts on top of the bar plots
        for i, fnmr, fmr in zip(x_axis, fnmrs, fmrs):
            plt.text(i - width / 2, fnmr + 1, str(round(fnmr, 1)), fontsize=10)
            plt.text(
                i - width / 2,
                fmr - 4.5,
                str(int(fmr)) if fmr <= 0.4 else str(round(abs(fmr), 1)),
                fontsize=10,
            )

    ax.set_axisbelow(True)

    # Plot finalization
    plt.title(
        f"FMR vs FNMR.  at Dev. FMR@{fmr_threshold*100}% under differente distances"
    )
    # ax.set_xlabel("Distances", fontsize=14)
    ax.set_ylabel(
        "FMR(%) vs FNMR(%)                               ", fontsize=14
    )
    plt.ylim([-25, 104])

    ax.set_xticks(X + 0.5)
    ax.set_xticklabels(distances, fontsize=16)

    yticks = np.array([-20, 0, 20, 40, 60, 80, 100])
    ax.set_yticks(yticks)
    ax.set_yticklabels(abs(yticks), fontsize=16)
    plt.axhline(0, linestyle="-", color="k")

    plt.legend()
    plt.grid()

    pdf.savefig(fig)
    pdf.close()
