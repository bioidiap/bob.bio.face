import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

import bob.measure

from bob.bio.base.score.load import get_split_dataframe


def gbu_report(scores_dev, output_filename, titles, figsize=(8, 6)):

    colors = plt.cm.tab20.colors

    # Plotting
    pdf = PdfPages(output_filename)

    # Figure for eval plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    for d_scores, title, color in zip(scores_dev, titles, colors):

        # Load the score files and fill in the angle associated to each camera
        impostors_dev, genuines_dev = get_split_dataframe(d_scores)

        # loading the dask dataframes
        i_dev = impostors_dev["score"].compute().to_numpy()
        g_dev = genuines_dev["score"].compute().to_numpy()

        fmr, fnmr = bob.measure.roc(i_dev, g_dev, n_points=40)
        # in %
        fnmr = 1 - fnmr
        fnmr *= 100
        fmr *= 100

        # plot.plot(roc_curve)
        plt.semilogx(fmr, fnmr, marker="o", label=title)
        pass

    pass

    # Plot finalization
    # plt.title(f"FMR vs FNMR .  at Dev. FMR@{fmr_threshold}")
    ax.set_xlabel("FMR%", fontsize=18)
    ax.set_ylabel("1 - FNMR%", fontsize=18)

    x_ticks = [0.0001, 0.01, 1, 100]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x}%" for x in x_ticks], fontsize=14)

    y_ticks = [0, 20, 40, 60, 80, 100]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{x}%" for x in y_ticks], fontsize=14)

    plt.legend()
    plt.grid()

    pdf.savefig(fig)
    pdf.close()
