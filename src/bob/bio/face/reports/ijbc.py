import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from bob.bio.base.score.load import cmc
from bob.measure import plot


def ijbc_report(scores_dev, output_filename, titles, figsize=(8, 6)):

    colors = plt.cm.tab10.colors

    # Plotting
    pdf = PdfPages(output_filename)

    # Figure for eval plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    for d_scores, title, color in zip(scores_dev, titles, colors):
        cmc_scores = cmc(d_scores)

        plot.detection_identification_curve(
            cmc_scores, rank=1, logx=True, color=color, label=title, linewidth=2
        )

        pass

    pass

    # Plot finalization
    ax.set_xlabel("False Positive Identification Rate", fontsize=18)
    ax.set_ylabel("True Positive Identification Rate", fontsize=18)

    x_ticks = [0.0001, 0.001, 0.01, 0.1, 1]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        ["$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$10^{0}$"],
        fontsize=14,
    )

    y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{round(y*100,0)}" for y in y_ticks], fontsize=14)

    plt.legend()
    plt.grid()

    pdf.savefig(fig)
    pdf.close()
