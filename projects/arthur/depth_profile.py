from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.reach import VisualOnly
from pixtools import clusters, utils

mice = [       
 ##  'HFR19',
    'HFR20',
 ##  'HFR21',
 ##  'HFR22',
 ##  'HFR23',
]

exp = Experiment(
    mice,
    VisualOnly,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/AZ_notes')

fig = clusters.depth_profile(exp, curated=True, in_brain=True)
plt.ylim([1750, -250])
plt.suptitle('Cluster depth profile')
utils.save(fig_dir / f'cluster_depth_profile')
