import json

import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush
from pixels.behaviours.pushpull import PushPull
from pixels.behaviours.reach import Reach
from pixels.behaviours.no_behaviour import NoBehaviour
from pixtools import utils

from setup import fig_dir

mthal = Experiment(
    [
        # paper behaviour + opto
        'C57_724',
        'C57_1288723',
        'C57_1288727',
        'C57_1313404',
        # muscimol spread test
        'C57_1318495',
        'C57_1318496',
        'C57_1318497',
        'C57_1318498',
    ],
    LeverPush,
    '~/duguidlab/thalamus_paper/Npx_data',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)


dirsens = Experiment(
    [
        # First cohort
        "C57_1319786",
        "C57_1319781",
        "C57_1319784",
        "C57_1319783",
        #"C57_1319782",
        #"C57_1319785",
        # Second cohort:
        "C57_1350950",
        "C57_1350951",
        "C57_1350952",
        #"C57_1350953",  # corrupted AP data
        "C57_1350954",
        "C57_1350955",
    ],
    PushPull,
    '~/duguidlab/Direction_Sensitivity/Data/Neuropixel',
)


reaching = Experiment(
    [
        "VR37",
        "VR40",
        "VR42",
    ],
    Reach,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

noise_tests = Experiment(
    ["noisetest1"],
    NoBehaviour,
    '~/duguidlab/visuomotor_control/neuropixels',
)

noise_test_unchanged = Experiment(
    ["noisetest_unchanged"],
    NoBehaviour,
    '~/duguidlab/visuomotor_control/neuropixels',
)

noise_test_nopi = Experiment(
    ["noisetest_nopi"],
    NoBehaviour,
    '~/duguidlab/visuomotor_control/neuropixels',
)

noise_test_no_caps = Experiment(
    ["VR50"],
    Reach,
    '~/duguidlab/visuomotor_control/neuropixels',
)

exps = {
    "mthal": mthal,
    "dirsens": dirsens,
    "reaching": reaching,
    "noise_test": noise_tests,
    "noise_test_unchanged": noise_test_unchanged,
    "noise_test_nopi": noise_test_nopi,
    "noise_test_no_caps": noise_test_no_caps,
}

noise = []
for name, exp in exps.items():
    for session in exp:
        for i in range(len(session.files)):
            path = session.processed / f"noise_{i}.json"
            with path.open() as fd:
                ses_noise = json.load(fd)
            date = datetime.datetime.strptime(session.name[:6], '%y%m%d')
            noise.append((session.name, date, name, ses_noise['median']))

df = pd.DataFrame(noise, columns=["session", "date", "project", "median SD"])

ax = sns.scatterplot(data=df, x="date", y="median SD", hue="project")
ax.set_ylabel('Median standard deviation of channels')
ax.set_xlabel('Recording date')
ax.set_title('Raw data standard deviation across recordings')
_, ymax = ax.get_ylim()
ax.set_ylim([0, ymax])
utils.save(fig_dir / f'noise_for_all_recordings')
