# Produces graphs of d' training progress in all mice across three month periods
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

sys.path.append("/home/s1735718/PixelsAnalysis/pixels-analysis")
sys.path.insert(0, "/home/s1735718/PixelsAnalysis/pixels/pixels")
from pixels.behaviours import Behaviour
from reach import Cohort
from pixels import ioutils
from pixtools import utils

#%%
# create the cohort class
cohort = Cohort.init_from_files(
    data_dir="/home/s1735718/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON",
    mouse_ids=[
        "VR46",
        "VR49",
        "VR52",
        "VR55",
        "VR59",
    ],
)

# now get the training results from file
results = pd.DataFrame(cohort.get_results())
trials = pd.DataFrame(cohort.get_trials())
# NB: D' may be incorrect where days have multiple sessions! May have to calculate manually.
# Or exclude from the plotting (CANT DO THIS FOR THE MAIN DISS ONLY TO SHOW TRAINING PROGRESSION)

#%%
# Plot linegraph of mouse and d' by day
plt.rcParams.update({"font.size": 35})
p = sns.lineplot(x="day", y="d_prime", hue="mouse_id", data=results, lw=3)

p.axhline(y=0, ls="--", color="grey", lw=3)
p.axhline(y=1.5, ls="--", color="green", lw=3)
plt.xlabel("Training Day")
plt.ylabel("d-Prime Score (d')")
plt.suptitle("Mouse d-Prime Score Throughout Training Period")

plt.gcf().set_size_inches(15,10)
utils.save(
    "/home/s1735718/Figures/cohort_dprime",
    nosize=True,
)
# %%
