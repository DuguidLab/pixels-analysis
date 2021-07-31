"""
Plot boxplot of proportion of responsive units in expert mice, aligning to
left visual stimulation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path

import numpy as np
import pandas as pd

from pixels import Experiment
from pixels.behaviours.reach import ActionLabels, Events, Reach
from pixtools import spike_rate, utils

mice = [
    # "HFR25",
    "HFR29",
]

exp = Experiment(
    mice,
    Reach,
    "~/duguidlab/visuomotor_control/neuropixels",
    "~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON",
)

fig_dir = Path("~/duguidlab/visuomotor_control/AZ_notes/npx-plots/expert")

rec_num = 0
#area = ["m2", "ppc"][rec_num]
area = ["right ppc"][rec_num] # for HFR29
duration = 2

# select units
units = exp.select_units(
    min_depth=0,
    max_depth=1200,
    # min_spike_width=0.4,
    name="cortex0-1200",
)

# define start, step, & end of confidence interval calculation, and the overlap between them.
start = 0.000
step = 0.200
end = 1.000
increment = 0.100

# get confidence interval for left & right visual stim.
cis_left0 = exp.get_aligned_spike_rate_CI(
    ActionLabels.miss_left,
    Events.led_on,
    start=start,
    step=step,
    end=end,
    bl_start=-1.000,
    bl_end=-0.050,
    units=units,
)
cis_left1 = exp.get_aligned_spike_rate_CI(
    ActionLabels.miss_left,
    Events.led_on,
    start=start+increment,
    step=step,
    end=end+increment,
    bl_start=-1.000,
    bl_end=-0.050,
    units=units,
)

data = []

for session in range(len(exp)):
    non_responsive = set()
    resps_set = set()

    for unit in units[session][rec_num]:
        ci = cis_left[session][rec_num][unit]

        # count total number of responsive units, aligning to ipsi & contra
        for bin in ci:
            ci_bin = ci[bin]
            if ci_bin[2.5] > 0:
                resps_set.add(unit)
                break
            elif ci_bin[97.5] < 0:
                resps_set.add(unit)
                break

    num_units = len(units[session][rec_num])
    num_resp = len(resps_set)
    data.append((num_resp, num_units, num_resp / num_units, area))

    print(exp[session].name, area)
    print('total number of units: ', num_units)
    print('total number of responsive: ', num_resp)
    print('proportion of responsive: ', num_resp / num_units)


_, axes = plt.subplots(2, 1)
name = exp[session].name
df = pd.DataFrame(
    data,
    columns=[
        "Number of Responsive Neurons",
        "Total Number of Neurons",
        "Proportion of Responsive Neurons",
        "Brain Area",
    ],
)
sns.boxplot(data=df, x="Brain Area", y="Total Number of Neurons", ax=axes[0])
sns.stripplot(
    data=df, x="Brain Area", y="Total Number of Neurons", color=".25", jitter=0,
    ax=axes[0],
)
sns.boxplot(data=df, x="Brain Area", y= "Number of Responsive Neurons", ax=axes[1])
sns.stripplot(
    data=df, x="Brain Area", y="Number of Responsive Neurons", color=".25", jitter=0,
    ax=axes[1],
)
plt.ylim(bottom=0)
utils.save(fig_dir / f"expert_Total_Number_of_Responsive_Neurons_rolling_bin_{name}_{area}.pdf")

plt.clf()
# plot proportion of responsive neurons
sns.boxplot(data=df, x="Brain Area", y="Proportion of Responsive Neurons")
sns.stripplot(
    data=df,
    x="Brain Area",
    y="Proportion of Responsive Neurons",
    color=".25",
    jitter=0,
)
plt.ylim(bottom=0)
utils.save(fig_dir / f"expert_Proportion_of_Responsive_Neurons_rolling_bin_{name}_{area}.pdf")
