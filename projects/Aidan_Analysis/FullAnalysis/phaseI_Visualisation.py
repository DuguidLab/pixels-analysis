# Import required packages
from argparse import Action
from base import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as sc
import sys
from channeldepth import meta_spikeglx
import statsmodels.api as sm
from statsmodels.formula.api import ols
from reach import Cohort
from xml.etree.ElementInclude import include
from matplotlib.pyplot import legend, title, ylabel, ylim
import matplotlib.lines as mlines
from tqdm import tqdm
from pixels import ioutils
from pixtools import utils
from textwrap import wrap

from functions import (
    event_times,
    unit_depths,
    per_trial_binning,
    event_times,
    within_unit_GLM,
    bin_data_average,
    # unit_delta, #for now use a local copy
)

#%%
# import datafile calculated previously
data = ioutils.read_hdf5("/data/aidan/concatenated_session_information.h5")
# test = ioutils.read_hdf5("/data/aidan/concatenated_session_information_test.h5")
# Add layer V data
layer_vals = pd.read_csv("granulardepths_VR46_VR59.csv")
#%%
# Visualise the data as a depth plot, showing units above and below layer V boundary for each session
fig, ax = plt.subplots(2, 1, sharex=True)
fig.set_size_inches(10, 10)

p = sns.swarmplot(
    data=data, x="depth", y="session", hue="Depth Classification", ax=ax[1]
)
p.set_xlim(0)
ax[1].legend(loc="upper right", bbox_to_anchor=(0, 1.7), title="Layer Class")

# Plot the overall distribution of points across depths.
q = sns.kdeplot(
    data=data, x="depth", ax=ax[0], hue="Depth Classification", legend=False
)
q.set_ylabel("")
q.set_yticklabels("")
q.set(yticks=[])

plt.subplots_adjust(wspace=0, hspace=0)
plt.suptitle("Distribution of Recorded Units Across pM2")

#%%
# Plot the proportion of each neuron type by up/downregulation
# As a stacked bar chart
# first isolate up/downreg units
delta_types = []
for i in data.iterrows():
    if i[1]["delta"] > 0:
        delta_types.append("upregulated")

    if i[1]["delta"] < 0:
        delta_types.append("downregulated")

data.insert(3, "delta type", delta_types)
# now plot these proportions, as a percentage
vals = data[["session", "delta type", "delta significance"]]
change = []
for i in vals.iterrows():
    if i[1]["delta significance"] == "non-significant":
        change.append("no-change")
    if (i[1]["delta significance"] == "significant") & (
        i[1]["delta type"] == "downregulated"
    ):
        change.append("downregulated")
    if (i[1]["delta significance"] == "significant") & (
        i[1]["delta type"] == "upregulated"
    ):
        change.append("upregulated")
vals["change"] = change
vals.drop(columns=["delta type", "delta significance"])

vals = (
    vals.groupby(["session", "change"])
    .size()
    .reset_index()
    .pivot(columns="change", index="session", values=0)
)
vals["upregulated"] = vals["upregulated"].fillna(0)
vals["downregulated"] = vals["downregulated"].fillna(0)
vals["no-change"] = vals["no-change"].fillna(0)
val_props = pd.DataFrame(index=vals.index)
val_props["upreg_%"] = (
    vals["upregulated"]
    / (vals["upregulated"] + vals["downregulated"] + vals["no-change"])
) * 100
val_props["downreg_%"] = (
    vals["downregulated"]
    / (vals["upregulated"] + vals["downregulated"] + vals["no-change"])
) * 100
val_props["nochange_%"] = (
    vals["no-change"]
    / (vals["upregulated"] + vals["downregulated"] + vals["no-change"])
) * 100
val_props.plot(kind="bar", stacked=True)

plt.suptitle("Proportion of Responsive Units in pM2")
utils.save(f"/home/s1735718/Figures/AllSessionUnitResponsiveness", nosize=True)

#%%
# Now shall plot delta firing rates by depth, as a violin plot
# NB: this may require binning depth to prevent pseudoreplication when collapsing across sessions
# Plot two seperate subplots for each session them merge, for up and downregulated data
pyr = data.loc[data["type"] == "pyramidal"]
inter = data.loc[data["type"] == "interneuron"]
pyr["all"] = 1
inter["all"] = 1
# first plot pyramidals
p = sns.violinplot(
    data=pyr,
    y="depth",
    x="session",
    hue="delta type",
    split=True,
    palette={"upregulated": "orange", "downregulated": "blue"},
)
for i, violin in enumerate(p.findobj(mpl.collections.PolyCollection)):
    violin.set_hatch("//")

p2 = sns.violinplot(
    data=inter,
    y="depth",
    x="session",
    hue="delta type",
    split=True,
    palette={"upregulated": "orange", "downregulated": "blue"},
)

plt.setp(p.collections, alpha=0.3)

# Now set legend
down = mpatches.Patch(facecolor="blue", alpha=0.5, label="Downregulated")

up = mpatches.Patch(facecolor="orange", alpha=0.5, label="Upregulated")

pyr_leg = mpatches.Patch(facecolor="grey", hatch="//", alpha=0.5, label="Pyramidals")

inter_leg = mpatches.Patch(facecolor="grey", alpha=0.5, label="Interneurons")
plt.legend(
    handles=[down, up, pyr_leg, inter_leg],
    bbox_to_anchor=(1, 1),
    title="Activity Change/Cell Type",
    fontsize=15,
)

plt.suptitle("Proportion of Significantly Changing Units During Grasp")
plt.gca().set_ylim(0, 1200)
plt.gca().invert_yaxis()
plt.gca().set_ylabel("pM2 Depth (um)")
plt.gca().set_xlabel("Session")
plt.xticks(rotation=90)


plt.show()


# Now plot the point at which layer V occurs

# %%
# A similar plot that may be of use is to instead plot a scatter with histogram overlaid
# Reuse scatter plot code, adjust it

for s, session in enumerate(myexp):
    name = session.name
    ses_data = data.loc[data["session"] == name]
    sigs = ses_data.loc[ses_data["delta significance"] == "significant"]
    nosigs = ses_data.loc[ses_data["delta significance"] == "non-significant"]

    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")

    # Finally, plot a graph of deltas relative to zero, by depth
    sns.scatterplot(
        x="delta",
        y="depth",
        data=sigs,
        # size="average firing rate",
        # sizes=(40, 400),
        s=100,
        linewidth=1,
        style="type",
        color="#972c7f",
        alpha=0.5,
        ax=ax,
        legend=False,
    )

    ######################################################################
    # ##Now overlay a kde plot over these points
    # ##Uncomment to plot this version
    # up_kde = sns.kdeplot(
    #     data=sigs.loc[sigs["delta type"] == "upregulated"],
    #     y="depth"
    # ).get_lines()[0].get_data()

    # #Extract density data, and scale it to the order of magnitude plotted (i.e., x10^5)
    # #Plot these values

    # up_kde=pd.DataFrame(up_kde, index=["density", "depth"])
    # up_kde.loc["density"] = up_kde.loc["density"].apply(lambda x: x*30000)
    # up_kde = up_kde.T
    # sns.scatterplot(data = up_kde, x="density", y = "depth")

    # down_kde = sns.kdeplot(
    #     data=sigs.loc[sigs["delta type"] == "downregulated"],
    #     y="depth"
    # ).get_lines()[1].get_data()

    # down_kde=pd.DataFrame(down_kde, index=["density", "depth"])
    # down_kde.loc["density"] = down_kde.loc["density"].apply(lambda x: x*-30000) #times by -1 to allow inverted plotting of kde plot
    # down_kde = down_kde.T
    # sns.scatterplot(data = down_kde, x="density", y = "depth")

    ######################################################################

    ##Uncomment below to plot histogram over points.
    # ax2=fig.add_subplot(111, label="2", frame_on=False)
    # ax3 = fig.add_subplot(111, label="2", frame_on=False)
    # up_hist = sns.histplot(
    #     data=sigs.loc[sigs["delta type"] == "upregulated"], y = "depth", binwidth=100, binrange=(0,1300), ax=ax2, legend=False
    # )

    # ax2.xaxis.tick_top()
    # ax2.set_xlim(-3,3)
    # ax2.axes.xaxis.set_visible(False)
    # ax2.axes.yaxis.set_visible(False)
    # ax2.set_ylabel("")

    # down_hist = sns.histplot(
    #     data=sigs.loc[sigs["delta type"] == "downregulated"], y = "depth", binwidth=100, binrange=(0,1300), ax=ax3, legend=False
    # )

    # ax3.invert_yaxis()
    # ax3.xaxis.tick_top()
    # ax3.set_xlim(3,-3)
    # ax3.axes.xaxis.set_visible(False)
    # ax3.axes.yaxis.set_visible(False)

    ######################################################################

    # p2 = sns.scatterplot(
    #     x="delta",
    #     y="depth",
    #     data=nosigs,
    #     size="average firing rate",
    #     sizes=(40, 400),
    #     s=100,
    #     linewidth=1,
    #     style="type",
    #     color="#221150",
    # )

    ax.axvline(x=0, color="green", ls="--")

    ax.set_xlim(-60, 60)
    ax.set_ylim(0, 1300)

    ax.set_xlabel("Δ Firing Rate (Hz)", fontsize=20)
    ax.set_ylabel("Depth of Recorded Neuron (μm)", fontsize=20)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.suptitle(
        "\n".join(
            wrap(
                f"Largest Trial Firing Rate Change by pM2 Depth - Session {name}",
                width=30,
            )
        ),
        y=1.1,
        fontsize=20,
    )

    # Create lengend manually
    # Entries for legend listed below as handles:
    sig_dot = mlines.Line2D(
        [],
        [],
        color="#221150",
        marker="o",
        linestyle="None",
        markersize=10,
        label="Significant",
    )
    nonsig_dot = mlines.Line2D(
        [],
        [],
        color="#972c7f",
        marker="o",
        linestyle="None",
        alpha=0.5,
        markersize=10,
        label="Non-Significant",
    )
    grey_dot = mlines.Line2D(
        [],
        [],
        color="grey",
        marker="o",
        linestyle="None",
        markersize=10,
        label="Pyramidal",
    )
    grey_cross = mlines.Line2D(
        [],
        [],
        color="grey",
        marker="X",
        lw=5,
        linestyle="None",
        markersize=10,
        label="Interneuronal",
    )

    plt.legend(
        handles=[  # sig_dot,
            # nonsig_dot,
            grey_dot,
            grey_cross,
        ],
        bbox_to_anchor=(1.7, 1),
        title="Significance (p < 0.05)",
        fontsize=15,
    )

    # Now invert y-axis and save
    ax.invert_yaxis()

    # utils.save(
    #     f"/home/s1735718/Figures/{name}_DeltaFR_byDepth",
    #     nosize=True,
    # )
    plt.show()

# %%
# Plot only the KDE values

for s, session in enumerate(myexp):
    name = session.name
    ses_data = data.loc[data["session"] == name]
    sigs = ses_data.loc[ses_data["delta significance"] == "significant"]
    nosigs = ses_data.loc[ses_data["delta significance"] == "non-significant"]

    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label = "2", frame_on=False)
    ######################################################################
    # ##Now overlay a kde plot over these points
    # ##Uncomment to plot this version
    up_kde = sns.kdeplot(
        data=sigs.loc[sigs["delta type"] == "upregulated"],
        y="depth", ax = ax, color = "blue"
    )

    down_kde = sns.kdeplot(
        data=sigs.loc[sigs["delta type"] == "downregulated"],
        y="depth", ax = ax2, color = "orange"
    )

    ax2.set_xlim(-0.002, 0.002)
    ax2.invert_xaxis()
    ax2.axes.xaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)


    plt.axvline(x=0, color="green", ls="--")
    ax.set_xlim(-0.002, 0.002)
    ax.set_ylim(0, 1300)
    ax.invert_yaxis()
    ax2.invert_yaxis()

    ax.set_xlabel("Likelihood of Unit Presence", fontsize=20)
    ax.set_ylabel("Depth Across pM2 (μm)", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_ticks(np.arange(0, 1300, 200))

    plt.suptitle(
        "\n".join(
            wrap(
                f"Kernel Density Estimate of Responsive Unit Type - Session {name}",
                width=30,
            )
        ),
        y=1.1,
        fontsize=20,
    )



    # Create lengend manually
    # Entries for legend listed below as handles:
    upreg = mlines.Line2D(
        [],
        [],
        color="blue",
        marker="o",
        linestyle="None",
        markersize=10,
        label="Upregulated Units",
    )

    downreg = mlines.Line2D(
        [],
        [],
        color="orange",
        marker="o",
        linestyle="None",
        markersize=10,
        label="Downregulated Units",
    )

    plt.legend(
        handles=[ 
            upreg,
            downreg,
        ],
        bbox_to_anchor=(1.7, 1),
        title="Significance (p < 0.05)",
        fontsize=15,
    )

    # Now invert y-axis and save
    plt.gca().invert_yaxis()

    # utils.save(
    #     f"/home/s1735718/Figures/{name}_DeltaFR_byDepth",
    #     nosize=True,
    # )
    plt.show()

# %%
# Furthermore, will now plot a ball and stick graph showing the actual changes in firing rates and the point at which these deltas occur
