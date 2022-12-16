import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages

import bob.measure

from bob.bio.base.score.load import get_split_dataframe


def mobio_report(
    scores_dev,
    scores_eval,
    output_filename,
    titles,
    fmr_threshold=1e-3,
    figsize=(16, 8),
    colors=plt.cm.tab10.colors,
):

    genders = [
        "m",
        "f",
    ]
    gender_labels = ["male", "female"]

    # colors = plt.cm.tab20.colors

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
        # Getting only the close distance to compute the threshold

        threshold = bob.measure.far_threshold(
            impostors_dev["score"].to_numpy(),
            genuines_dev["score"].to_numpy(),
            far_value=fmr_threshold,
        )

        def compute_fmr_fnmr(impostor_scores, genuine_scores):
            eval_fmr, eval_fnmr = bob.measure.farfrr(
                impostor_scores["score"].to_numpy(),
                genuine_scores["score"].to_numpy(),
                threshold,
            )
            return eval_fmr, eval_fnmr

        # Combined
        eval_fmr_fnmr[title].append(
            compute_fmr_fnmr(impostors_eval, genuines_eval)
        )

        for gender in genders[1:]:

            i_eval = impostors_eval.loc[impostors_eval.probe_gender == gender]
            g_eval = genuines_eval.loc[genuines_eval.probe_gender == gender]

            eval_fmr_fnmr[title].append(compute_fmr_fnmr(i_eval, g_eval))

    pass

    # Plotting
    pdf = PdfPages(output_filename)

    # Figure for eval plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    width = 0.8 / len(titles)
    X = np.arange(len(genders))

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
            plt.text(
                i - width / 2, fnmr + 0.8, str(round(fnmr, 2)), fontsize=12
            )
            # print(i - width / 2)
            plt.text(
                i - width / 2, fmr - 2.2, str(round(abs(fmr), 2)), fontsize=12
            )

    ax.set_axisbelow(True)

    # Plot finalization
    plt.title(f"FMR vs FNMR .  at Dev. FMR@{fmr_threshold*100}%")
    # ax.set_xlabel("Genders", fontsize=18)
    ax.set_ylabel("FMR(%) vs FNMR(%)                        ", fontsize=15)
    # plt.ylim([-5, 40])

    ax.set_xticks(X + 0.5)
    ax.set_xticklabels(gender_labels, fontsize=14)

    yticks = np.array([-5, 0, 5, 15, 25, 35])
    ax.set_yticks(yticks)
    ax.set_yticklabels(abs(yticks), fontsize=16)

    plt.axhline(0, linestyle="-", color="k")

    plt.legend()
    plt.grid()

    pdf.savefig(fig)
    pdf.close()
