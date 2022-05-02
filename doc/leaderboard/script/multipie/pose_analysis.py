# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: bob_pipelines
#     language: python
#     name: bob_pipelines
# ---

import os

import matplotlib as mpl
import numpy as np
import pandas as pd

import bob.measure

mpl.rcParams.update({"font.size": 14})
import matplotlib.pyplot as plt

# %matplotlib inline

# ### Select baselines

baselines = {
    "FaceNet (D. Sandberg)": "facenet-sanderberg",
    "Casia-Webface - Inception Resnet v1": "inception-resnetv1-casiawebface",
    "Casia-Webface - Inception Resnet v2": "inception-resnetv2-casiawebface",
    "MSCeleb - Inception Resnet v1": "inception-resnetv1-msceleb",
    "MSCeleb - Inception Resnet v2": "inception-resnetv2-msceleb",
}

# ### Define some utilitary functions

# +
results_dir = "./results/"

# Function to load the scores in CSV format
def load_scores(baseline, protocol, group):
    scores = pd.read_csv(
        os.path.join(
            results_dir,
            "multipie_{}".format(protocol),
            baseline,
            "scores-{}.csv".format(group),
        )
    )
    return scores


# Function to separate genuines from impostors
def split(df):
    impostors = df[df["probe_reference_id"] != df["bio_ref_reference_id"]]
    genuines = df[df["probe_reference_id"] == df["bio_ref_reference_id"]]
    return impostors, genuines


# -

# ### Establish Camera to Angle conversion

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

# ### Run pose analysis and plot results

# +
# Figure for Dev plot
fig1 = plt.figure(figsize=(8, 6))
# Figure for Eval plot
fig2 = plt.figure(figsize=(8, 6))
for name, baseline in baselines.items():

    # Load the score files and fill in the angle associated to each camera
    dev_scores = load_scores(baseline, "P", "dev")
    eval_scores = load_scores(baseline, "P", "eval")
    dev_scores["angle"] = dev_scores["probe_camera"].map(camera_to_angle)
    eval_scores["angle"] = eval_scores["probe_camera"].map(camera_to_angle)

    angles = []
    dev_hters = []
    eval_hters = []
    # Run the analysis per view angle
    for (angle, dev_df), (_, eval_df) in zip(
        dev_scores.groupby("angle"), eval_scores.groupby("angle")
    ):
        # Separate impostors from genuines
        dev_impostors, dev_genuines = split(dev_df)
        eval_impostors, eval_genuines = split(eval_df)

        # Compute the min. HTER threshold on the Dev set
        threshold = bob.measure.min_hter_threshold(
            dev_impostors["score"], dev_genuines["score"]
        )

        # Compute the HTER for the Dev and Eval set at this particular threshold
        dev_far, dev_frr = bob.measure.farfrr(
            dev_impostors["score"], dev_genuines["score"], threshold
        )
        eval_far, eval_frr = bob.measure.farfrr(
            eval_impostors["score"], eval_genuines["score"], threshold
        )
        angles.append(angle)
        dev_hters.append(1 / 2 * (dev_far + dev_frr))
        eval_hters.append(1 / 2 * (eval_far + eval_frr))

    # Update plots
    plt.figure(1)
    plt.plot(angles, dev_hters, label=name, marker="x")

    plt.figure(2)
    plt.plot(angles, eval_hters, label=name, marker="x")

# Plot finalization
plt.figure(1)
plt.title("Dev. min. HTER")
plt.xlabel("Angle")
plt.ylabel("Min HTER")
plt.legend()
plt.grid()

plt.figure(2)
plt.title("Eval. HTER @Dev min. HTER")
plt.xlabel("Angle")
plt.ylabel("HTER")
plt.legend()
plt.grid()
