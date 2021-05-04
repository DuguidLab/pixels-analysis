import json

import matplotlib.pyplot as plt
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush
from pixels.behaviours.pushpull import PushPull
from pixels.behaviours.reach import Reach


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
        "C57_1319786",
        "C57_1319781",
        "C57_1319784",
        "C57_1319783",
        #"C57_1319782",
        #"C57_1319785",
    ],
    PushPull,
    '~/duguidlab/Direction_Sensitivity/Data/Neuropixel',
)


vis = Experiment(
    [
        "VR17",
        "VR18",
        "VR16",
        #"HFR20",
        #"HFR21",
        #"HFR19",
        #"HFR22",
        #"HFR23",
    ],
    Reach,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

exps = [mthal, dirsens, vis]

noise = []
for exp in exps:
    for session in exp:
        for i in range(len(session.files)):
            path = session.processed / f"noise_{i}.json"
            with path.open() as fd:
                ses_noise = json.load(fd)
            noise.append((session.name, ses_noise['median']))

print(noise)

#noise.sort(key=lambda t: t[0])
#medians = [n for _, n in noise]
#
#plt.plot(medians)
#ax = plt.gca()
#ax.set_ylabel('Standard deviation')
#ax.set_xlabel('Recording no.')
#ax.set_title('Raw data standard deviation across recordings')
#plt.show()