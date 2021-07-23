from pathlib import Path

import pandas as pd
import numpy as np

from pixels import Experiment
from pixels.behaviours.reach import Reach, ActionLabels, Events
from pixtools import spike_rate, utils

mice = [       
    'HFR25',
    #'HFR29', # right PPC recording only
]

exp = Experiment(
    mice,
    Reach,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

# Envelope for plots, 95% confidence interval
ci = 95

## Select units
duration = 2
units = exp.select_units(
        min_depth=0, max_depth=1200,
        #min_spike_width=0.4,
        name="cortexs0-1200"
        )

# get spike rate for left & right visual stim.
stim_left = exp.align_trials(
    ActionLabels.miss_left,
    Events.led_on,
    'spike_rate',
    units=units,
    duration=duration,
)

# side of PPC recording
sides = [
    "left",
    "right",
]

ipsi_ppc_list = []
contra_m2_list = []
contra_ppc_list = []

for session in range(len(exp)):
    # for naive m2
    contra_m2_list.append(stim_left[session][0])

    # for ppc
    if sides[session] == "left":
        ipsi_ppc_list.append(stim_left[session][1])
    else:
        contra_ppc_list.append(stim_left[session][1])

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
