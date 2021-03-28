import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import utils

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

sns.set(font_scale=0.4)
fig_dir = '~/duguidlab/visuomotor_control/figures'
duration = 2

_, axes = plt.subplots(len(exp), 1)

hits = exp.align_trials(
    ActionLabels.cued_shutter_push_full,
    Events.back_sensor_open,
    'spike_rate',
    duration=duration,
    min_depth=500,
    max_depth=1200,
)

for session in range(len(exp)):
    name = exp[session].name
    data = hits[session].stack()
    norm = StandardScaler().fit(data).transform(data)
    num_components = norm.shape[1]
    pca = PCA(n_components=num_components)
    # probably don't need to transform
    pc = pca.fit(norm).transform(norm)

    axes[session].plot(np.cumsum(pca.explained_variance_ratio_))
    axes[session].plot([0, num_components], [0, 1], '--', linewidth=0.4)
    plt.xlabel('Components')
    plt.ylabel('Cumulative explained variance')

utils.save(fig_dir / f"scree_plot_good_deep_units_{duration}s.png")
