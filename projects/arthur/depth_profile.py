"""
Plot ipsi & contralateral visual stimulation responsive units across all layers, with cell type specified.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from pixels import Experiment
from pixels.behaviours.reach import Reach, VisualOnly, ActionLabels, Events
from pixtools import spike_rate, clusters, utils

mice = [       
    #'HFR20', #naive
    #'HFR22', #naive
    #'HFR23', #naive
    'HFR25', #trained
    'HFR29', #trained
]   

exp = Experiment(
    mice,
    VisualOnly,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

rec_num = 0
duration = 2
sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/AZ_notes/npx-plots/expert')

fig = clusters.depth_profile(exp, curated=True, group=None, in_brain=True)
plt.ylim([1750, -250])
plt.suptitle('Cluster depth profile')
plt.gcf().set_size_inches(10, 5)
utils.save(fig_dir / f'21072021cluster_depth_profile_expert', nosize=True)
