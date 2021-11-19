import os

os.environ['KILOSORT3_PATH'] = '/opt/neuropixels/Kilosort'

from pathlib import Path

from pixels import Experiment
from pixels.behaviours.no_behaviour import NoBehaviour

fig_dir = Path('~/duguidlab/visuomotor_control/figures')

mice = [
    #"noisetest1",
    "noisetest_unchanged",
    "noisetest_nopi",
]

exp = Experiment(
    mice,
    NoBehaviour,
    '~/duguidlab/visuomotor_control/neuropixels',
)

exp.assess_noise()
