"""
Ready-made ipsi & contra spike rate df for naive mice.

concatenate left & right PPC recording sessions (that aligning to stim_left
& stim_right respectively) seperately, and create ipsilateral & contralateral
alignment sessions. 

in naive group, all M2 recordings were on the left. i.e., stim_left alignment
is ipsi, & stim_right alignment is contra for M2.

currently only one trained mouse, HFR25, has simultaneous right M2 & left+right
PPC recording. thus, for m2, stim_left is contra, stim_right is ipsi.

====
In naive mice, given left M2 recording:
    ipsi_m2: left visual stim. alignment
    contra_m2: right visual stim. alignment

In trained mice, given right M2 recording:
    ipsi_m2: right visual stim. alignment
    contra_m2: left visual stim. alignment

In both groups:
    ipsi_ppc: 
        left ppc recordings & left visual stim. alignment;
        right ppc recordings & right visual stim. alignment.

    contra_ppc:
        left ppc recordings & right visual stim. alignment;
        right ppc recordings & left visual stim. alignment.
===
usage: from naive_ipsi_contra_spike_rate import *

IMPORTANT NOTE: IPSI & CONTRA CANNOT BE USED FOR CORRELATION AS THE NUMBER OF
TRIALS DO NOT MATCH CORRESPONDINGLY.

In naive group, M2 recordings are all performed on the left, and PPC recordings
are bilateral. Thus, only left PPC recordings has the number of trials matches
with M2, the ones of right PPC recording are swapped.  Use 'stim_left' &
'stim_right' instead, and note down ipsi/contra manually. 
"""

from pathlib import Path

import pandas as pd
import numpy as np

from pixels import Experiment
from pixels.behaviours.reach import VisualOnly, ActionLabels, Events
from pixtools import spike_rate, utils

mice = [       
    'HFR20',
    'HFR22',
    'HFR23',
]

exp = Experiment(
    mice,
    VisualOnly,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

fig_dir = Path('~/duguidlab/visuomotor_control/AZ_notes/npx-plots/naive')

# Envelope for plots, 95% confidence interval
ci = 95

## Select units
duration = 2
units = exp.select_units(
        min_depth=0, max_depth=1200,
        #min_spike_width=0.4,
        name="cortex0-1200"
        )

# get spike rate for left & right visual stim.
stim_left = exp.align_trials(
    ActionLabels.naive_left,
    Events.led_on,
    'spike_rate',
    units=units,
    duration=duration,
)
stim_right = exp.align_trials(
    ActionLabels.naive_right,
    Events.led_on,
    'spike_rate',
    units=units,
    duration=duration,
)

# side of PPC recording
sides = [
    "left",
    "right",
    "left",
    "right",
    "left",
    "right",
]

ipsi_m2_list = []
ipsi_ppc_list = []
contra_m2_list = []
contra_ppc_list = []

for session in range(len(exp)):
    # for naive m2
    ipsi_m2_list.append(stim_left[session][0])
    contra_m2_list.append(stim_right[session][0])

    # for ppc
    if sides[session] == "left":
        ipsi_ppc_list.append(stim_left[session][1])
        contra_ppc_list.append(stim_right[session][1])
    else:
        contra_ppc_list.append(stim_left[session][1])
        ipsi_ppc_list.append(stim_right[session][1])

ipsi_m2 = pd.concat(
	ipsi_m2_list, axis=1, copy=False,
	keys=range(len(ipsi_m2_list)),
	names=["session", "unit", "trial"],
)
ipsi_ppc = pd.concat(
	ipsi_ppc_list, axis=1, copy=False,
	keys=range(len(ipsi_ppc_list)),
	names=["session", "unit", "trial"],
)
contra_m2 = pd.concat(
	contra_m2_list, axis=1, copy=False,
	keys=range(len(contra_m2_list)),
	names=["session", "unit", "trial"],
)
contra_ppc = pd.concat(
    contra_ppc_list, axis=1, copy=False,
	keys=range(len(contra_ppc_list)),
	names=["session", "unit", "trial"],
)
ipsi = pd.concat([ipsi_m2, ipsi_ppc], axis=1, keys=['m2', 'ppc'], names=['area'])
contra = pd.concat([contra_m2, contra_ppc], axis=1, keys=['m2', 'ppc'], names=['area'])
