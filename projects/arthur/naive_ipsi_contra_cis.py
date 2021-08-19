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
from pixtools import spike_rate, utils, rolling_bin


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

cis_left = rolling_bin.get_rolling_bins(
    exp=exp,
    units=units,
    al=ActionLabels.naive_left,
    ci_s=start,
    step=step,
    ci_e=end,
    bl_s=-1.000,
    bl_e=-0.050,
    increment=0.100,
)

cis_right = rolling_bin.get_rolling_bins(
    exp=exp,
    units=units,
    al=ActionLabels.naive_right,
    ci_s=start,
    step=step,
    ci_e=end,
    bl_s=-1.000,
    bl_e=-0.050,
    increment=0.100,
)

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
