import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools.utils import subplots2d


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

def save(name):
    plt.gcf().savefig(fig_dir / name, bbox_inches='tight', dpi=300)

duration = 4


def scree(data, ax):
    norm = StandardScaler().fit(data).transform(data)
    num_components = norm.shape[1]
    pca = PCA(n_components=num_components)
    # probably don't need to transform
    pc = pca.fit(norm).transform(norm)

    ax.plot(np.cumsum(pca.explained_variance_ratio_))
    ax.plot([0, num_components], [0, 1], '--', linewidth=0.4)
    plt.xlabel('Components')
    plt.ylabel('Cumulative explained variance')


_, axes = plt.subplots(3, len(exp))

brain = exp.align_trials(
    ActionLabels.cued_shutter_push_full,
    Events.back_sensor_open,
    'spike_rate',
    duration=duration,
)

d1200 = exp.align_trials(
    ActionLabels.cued_shutter_push_full,
    Events.back_sensor_open,
    'spike_rate',
    duration=duration,
    min_depth=500,
    max_depth=1200,
)

d900 = exp.align_trials(
    ActionLabels.cued_shutter_push_full,
    Events.back_sensor_open,
    'spike_rate',
    duration=duration,
    min_depth=500,
    max_depth=900,
)

for session in range(len(exp)):
    scree(brain[session].stack(), axes[0][session])
    name = exp[session].name
    scree(d1200[session].stack(), axes[1][session])
    scree(d900[session].stack(), axes[2][session])
save(f"scree_plot.png")
