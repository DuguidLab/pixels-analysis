"""
Ready-made ipsi & contra confidence intervals df for naive mice.

usage:
from naive_ipsi_contra_cis import *
"""
from pathlib import Path

import numpy as np
import pandas as pd

from pixels import Experiment
from pixels.behaviours.reach import ActionLabels, Events, VisualOnly
from pixtools import spike_rate, utils

def rename_bin(df,l, names):
    """
    rename bins of the ci dataframe into their starting timestamps, i.e.,
    if bin=100, this bin starts at 100ms (aligning to the start of ci calculation).
    ====
    parameters:

    df: pd dataframe that contains cis.

    l: level of the bin in the dataframe.

    names: list of numbers/strings that will be the bin's new names.
    """
    new_names_tuple = dict(zip(df.columns.levels[l], names))
    df = df.rename(columns=new_names_tuple, level=l)

    return df

mice = [       
    "HFR20",
    "HFR22",
    "HFR23",
]

exp = Experiment(
    mice,
    VisualOnly,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

fig_dir = Path('~/duguidlab/visuomotor_control/AZ_notes/npx-plots/naive')

duration = 2
ci = 95

#exp.set_cache(False)
# select units
units = exp.select_units(
    min_depth=0,
    max_depth=1200,
    #min_spike_width=0.4,
    name="cortex0-1200",
)

# define start, step, & end of confidence interval bins
start = 0.000
step = 0.200
end = 1.000
increment = 0.100

# get confidence interval for left & right visual stim.
print('get cis...')
cis_left0 = exp.get_aligned_spike_rate_CI(
    ActionLabels.naive_left,
    Events.led_on,
    start=start,
    step=step,
    end=end,
    bl_start=-1.000,
    bl_end=-0.050,
    units=units,
)
cis_left1 = exp.get_aligned_spike_rate_CI(
    ActionLabels.naive_left,
    Events.led_on,
    start=start+increment,
    step=step,
    end=end+increment,
    bl_start=-1.000,
    bl_end=-0.050,
    units=units,
)

cis_left0 = rename_bin(df=cis_left0, l=3, names=[0, 200, 400, 600, 800])
cis_left1 = rename_bin(df=cis_left1, l=3, names=[100, 300, 500, 700, 900])
cis_left = pd.concat([cis_left0, cis_left1], axis=1)

cis_right0 = exp.get_aligned_spike_rate_CI(
    ActionLabels.naive_right,
    Events.led_on,
    start=start,
    step=step,
    end=end,
    bl_start=-1.000,
    bl_end=-0.050,
    units=units,
)
cis_right1 = exp.get_aligned_spike_rate_CI(
    ActionLabels.naive_right,
    Events.led_on,
    start=start+increment,
    step=step,
    end=end+increment,
    bl_start=-1.000,
    bl_end=-0.050,
    units=units,
)
cis_right0 = rename_bin(df=cis_right0, l=3, names=[0, 200, 400, 600, 800])
cis_right1 = rename_bin(df=cis_right1, l=3, names=[100, 300, 500, 700, 900])
cis_right = pd.concat([cis_right0, cis_right1], axis=1)

# side of the PPC recording
sides = [
    'left',
    'right',
    'left',
    'right',
    'left',
    'right',
]
              
ipsi_m2_list = []
contra_m2_list = []
ipsi_ppc_list = []
contra_ppc_list = []

for session in range(len(exp)):
    # m2
    print(exp[session].name)

    ipsi_m2_list.append(cis_left[session][0])
    contra_m2_list.append(cis_right[session][0])
    # ppc
    if sides[session] == 'left':
        ipsi_ppc_list.append(cis_left[session][1])
        contra_ppc_list.append(cis_right[session][1])
    else:
        contra_ppc_list.append(cis_left[session][1])
        ipsi_ppc_list.append(cis_right[session][1])

ipsi_m2_ci = pd.concat(
    ipsi_m2_list, axis=1, copy=False,
    keys=range(len(ipsi_m2_list)),
    names=['session', 'unit', 'bin']
    )
contra_m2_ci = pd.concat(
    contra_m2_list, axis=1, copy=False,
    keys=range(len(contra_m2_list)),
    names=['session', 'unit', 'bin']
)
ipsi_ppc_ci = pd.concat(
    ipsi_ppc_list, axis=1, copy=False,
    keys=range(len(ipsi_ppc_list)),
    names=['session', 'unit', 'bin']
    )
contra_ppc_ci = pd.concat(
    contra_ppc_list, axis=1, copy=False,
    keys=range(len(contra_ppc_list)),
    names=['session', 'unit', 'bin']
)

ipsi_ci = pd.concat(
    [ipsi_m2_ci, ipsi_ppc_ci],
    axis=1, copy=False,
    keys=['m2', 'ppc'],
    names=['area']
)
contra_ci = pd.concat(
    [contra_m2_ci, contra_ppc_ci],
    axis=1, copy=False,
    keys=['m2', 'ppc'],
    names=['area']
)

ipsi_ci = pd.concat(
	[ipsi_m2_ci, ipsi_ppc_ci],
	axis=1, copy=False,
	keys=['m2', 'ppc'],
	names=['area']
)
contra_ci = pd.concat(
	[contra_m2_ci, contra_ppc_ci],
	axis=1, copy=False,
	keys=['m2', 'ppc'],
	names=['area']
)
