import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush
from pixtools import clusters


mice = [
    'C57_724',
    'C57_1288723',
    'C57_1288727',
]

exp = Experiment(
    mice,
    LeverPush,
    '~/duguidlab/thalamus_paper/Npx_data',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

exp.set_cache(True)
sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/figures').expanduser()


# all clusters
#fig = clusters.depth_profile(exp, curated=False)
#plt.ylim([-250, 4000])
#plt.suptitle('Cluster depth profile uncurated')
#fig.savefig(fig_dir / f'cluster_depth_profile_uncurated.png', bbox_inches='tight', dpi=300)
#
#fig = clusters.depth_profile(exp, curated=True)
#plt.ylim([-250, 4000])
#plt.suptitle('Cluster depth profile curated')
#fig.savefig(fig_dir / f'cluster_depth_profile_curated.png', bbox_inches='tight', dpi=300)


# post-curation good units in the brain
fig = clusters.depth_profile(exp, curated=True, group='good', in_brain=True)
plt.suptitle('Cluster depth profile good units in brain')
plt.ylim([2000, 0])
fig.savefig(fig_dir / f'cluster_depth_profile_neurons.png', bbox_inches='tight', dpi=300)
#plt.show()
