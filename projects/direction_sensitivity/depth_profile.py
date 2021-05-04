from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.pushpull import PushPull
from pixtools import clusters, utils

mice = [
    "C57_1319786",
    "C57_1319781",
    "C57_1319784",
    "C57_1319783",
    "C57_1319782",
    "C57_1319785",
]

exp = Experiment(
    mice,
    PushPull,
    '~/duguidlab/Direction_Sensitivity/Data/Neuropixel',
    #'~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/figures/DS')

# all clusters
fig = clusters.depth_profile(exp, curated=False)
plt.ylim([2000, - 250])
plt.suptitle('Cluster depth profile uncurated')
utils.save(fig_dir / f'cluster_depth_profile_uncurated')

#fig = clusters.depth_profile(exp, curated=True)
#plt.ylim([-250, 2000])
#plt.suptitle('Cluster depth profile curated')
#utils.save(fig_dir / f'cluster_depth_profile_curated')

# post-curation good units in the brain
#fig = clusters.depth_profile(exp, curated=True, group='good')
#plt.suptitle('Cluster depth profile good units in brain')
#plt.ylim([2000, 0])
#utils.save(fig_dir / f'cluster_depth_profile_neurons')
