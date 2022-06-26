#%%
##This script simply produces a pie chart of the distribution of neuron types
# Import required packages
import enum
from xml.etree.ElementInclude import include
from cv2 import multiply, rotate
from matplotlib.pyplot import colormaps, title, ylabel, ylim
import numpy as np
import pandas as pd
import seaborn as sns

import sys
from base import *
from channeldepth import *
from functions import event_times, per_trial_spike_rate
from functions import unit_depths

sys.path.append("/home/s1735718/PixelsAnalysis/pixels-analysis")
sys.path.insert(0, "/home/s1735718/PixelsAnalysis/pixels/pixels")
from pixtools.utils import Subplots2D  # use the local copy of base.py
from pixtools import utils
from pixtools import spike_rate
from pixels import ioutils
from pixels import PixelsError
from matplotlib.axes import Axes

##Now select units for pyramidal and interneurons
all_units = myexp.select_units(group="good", name="m2", max_depth=1200, min_depth=200)

# select all pyramidal neurons
pyramidal_units = get_pyramidals(myexp)

# and all interneurons
interneuron_units = get_interneurons(myexp)


# Then align trials
all_aligned = myexp.align_trials(
    ActionLabels.correct, Events.led_off, "spike_rate", duration=10, units=all_units
)

pyramidal_aligned = myexp.align_trials(
    ActionLabels.correct,
    Events.led_off,
    "spike_rate",
    duration=10,
    units=pyramidal_units,
)

interneuron_aligned = myexp.align_trials(
    ActionLabels.correct,
    Events.led_off,
    "spike_rate",
    duration=10,
    units=interneuron_units,
)

#%%
# Now plot a chart containing a count of all neuron types per session
# To do so, will require a count of neuronal units, only need to do this once
ses_counts = []
raw_data = []
for s, session in enumerate(myexp):
    name = session.name

    ses_pyramidals = pyramidal_aligned[s]
    ses_interns = interneuron_aligned[s]

    pyr_count = pd.DataFrame(
        [
            name,
            "pyramidal",
            len(ses_pyramidals.columns.get_level_values("unit").unique()),
        ]
    ).T

    pyr_num = pd.DataFrame(
        [
            name,
            "pyramidal",
            ses_pyramidals.columns.get_level_values("unit").unique(),
        ]
    ).T
    ses_counts.append(pyr_count)
    raw_data.append(pyr_num)

    int_count = pd.DataFrame(
        [
            name,
            "interneuron",
            len(ses_interns.columns.get_level_values("unit").unique()),
        ]
    ).T

    int_num = pd.DataFrame(
        [
            name,
            "interneuron",
            ses_interns.columns.get_level_values("unit").unique(),
        ]
    ).T

    ses_counts.append(int_count)
    raw_data.append(int_num)

ses_counts = pd.concat(ses_counts)
ses_counts.rename(columns={0: "session", 1: "neuron", 2: "count"}, inplace=True)

ses_raw_data = pd.concat(raw_data)
ses_raw_data.rename(columns={0: "session", 1: "neuron", 2: "unit"}, inplace=True)
ses_raw_data = ses_raw_data.explode("unit").reset_index(drop=True)
#%%
# plot these as stacked bar
# plt.rcParams.update({"font.size": 30})

# plot_dat = (
#     ses_raw_data.groupby(["session", "neuron"])
#     .size()
#     .reset_index()
#     .pivot(columns="neuron", index="session")
# )
# plot_dat.plot(kind="bar", stacked=True, color=["pink", "cornflowerblue"])
# plt.legend(
#     labels=["Interneuron", "Pyramidal"],
#     bbox_to_anchor=(1.22, 1.1),
#     title="Neuronal Type",
# )

# plt.suptitle("Proportion of Neuronal Types Within pM2")
# plt.xlabel("Recording Session & Mouse ID (SessionDate_MouseID)")
# plt.ylabel("Number of Recorded Neurons")
# plt.xticks(rotation=45)
# plt.gcf().set_size_inches(10, 10)


# Uncomment below to save as pdf
# utils.save(
#     f"/home/s1735718/Figures/NeuronalTypeCountPlot",
#     nosize=True,
# )


#%%

####Will also plot a depth graph of neuron type####
####Like the previous graph but Y is depth rather than count, x is session again####
##Get all unit depths for the experiment
depths = unit_depths(myexp)
all_unit_depths = []

for s, session in enumerate(myexp):
    name = session.name
    ses_depths = depths[name]
    ses_pyramidals = pyramidal_aligned[s]
    ses_int = interneuron_aligned[s]

    pyr_depths = ses_depths[ses_pyramidals.columns.get_level_values("unit").unique()]
    int_depths = ses_depths[ses_int.columns.get_level_values("unit").unique()]

    # melt then merge these datasets to allow plotting
    pyr_depths = pyr_depths.melt()
    pyr_depths["type"] = "pyramidal"

    int_depths = int_depths.melt()
    int_depths["type"] = "interneuron"

    units = pd.concat([pyr_depths, int_depths]).reset_index(drop=True)
    units["session"] = name

    all_unit_depths.append(units)

all_unit_depths = pd.concat(all_unit_depths, axis=0).reset_index(drop=True)
# %%
# Plot these depths by session and type
plt.rcParams.update({"font.size": 50})

p = sns.stripplot(
    x="session",
    y="value",
    hue="type",
    data=all_unit_depths,
    s=15,
    linewidth=1,
    palette="bright",
)

plt.xticks(rotation=90)
plt.xlabel("Recording Session")
plt.ylabel("pM2 Depth (μm)")
plt.ylim(0)
plt.gca().invert_yaxis()  # invert the y axis, make understanding easier


# Plot lines showing lower bound of pM2

plt.axhline(y=1200, color="black", ls="--")
plt.text(1.55, 1200, "pM2 Lower Bound")

plt.legend(bbox_to_anchor=(1, 1), title="Neuron Type")
plt.gcf().set_size_inches(15, 15)
plt.suptitle("Neuronal Types by Location in pM2")

# Save Figure then Show
utils.save(
    "/home/s1735718/Figures/neurontype_by_depth",
    nosize=True,
)

plt.show()

# %%
