#%%
# First import required packages
from argparse import Action
from base import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as sc
import sys
from channeldepth import meta_spikeglx
import statsmodels.api as sm
from statsmodels.formula.api import ols
from reach import Cohort
from xml.etree.ElementInclude import include
from matplotlib.pyplot import legend, title, ylabel, ylim
import matplotlib.lines as mlines
from tqdm import tqdm
from pixels import ioutils

from functions import (
    event_times,
    unit_depths,
    per_trial_binning,
    event_times,
    within_unit_GLM,
    bin_data_average,
    # unit_delta, #for now use a local copy
)

#%%
# First select units based on spike width analysis
all_units = myexp.select_units(group="good", name="m2", min_depth=200, max_depth=1200)

pyramidals = get_pyramidals(myexp)
interneurons = get_interneurons(myexp)


# Then import the known neuronal type distributions from csv
known_units = pd.read_csv("cell_atlas_neuron_distribution_M2.csv", index_col=False)

#%%
# Now align Trials
per_unit_aligned = myexp.align_trials(
    ActionLabels.correct, Events.led_off, "spike_rate", duration=2, units=all_units
)

left_aligned = myexp.align_trials(
    ActionLabels.correct_left, Events.led_off, "spike_rate", duration=2, units=all_units
)

right_aligned = myexp.align_trials(
    ActionLabels.correct_right,
    Events.led_off,
    "spike_rate",
    duration=2,
    units=all_units,
)

pyramidal_aligned = myexp.align_trials(
    ActionLabels.correct, Events.led_off, "spike_rate", duration=2, units=pyramidals
)
interneuron_aligned = myexp.align_trials(
    ActionLabels.correct, Events.led_off, "spike_rate", duration=2, units=interneurons
)

# Now compare counts of neuropixel data to known data
counts = pd.DataFrame(columns=["Session", "Neurons", "Excitatory", "Inhibitory"])
for s, session in enumerate(myexp):

    name = session.name
    ses_pyr_count = len(pyramidals[s])
    ses_int_count = len(interneurons[s])
    total = ses_pyr_count + ses_int_count
    neurons = pd.Series(
        [name, total, ses_pyr_count, ses_int_count], index=counts.columns
    )

    counts.loc[s] = neurons  # append as new row

# Now calculate percentage proportions
counts["excitatory_percentage"] = (counts.Excitatory / counts.Neurons) * 100
counts["inhibitory_percentage"] = (counts.Inhibitory / counts.Neurons) * 100
median_counts = counts.median()

known_counts = known_units[["Neurons", "Excitatory", "Inhibitory"]]
known_counts["excitatory_percentage"] = (
    known_counts.Excitatory / known_counts.Neurons
) * 100
known_counts["inhibitory_percentage"] = (
    known_counts.Inhibitory / known_counts.Neurons
) * 100
known_median = known_counts.median()

# Now perform a chi-squared analysis to check across distributions
sc.chisquare(
    median_counts[["excitatory_percentage", "inhibitory_percentage"]],
    f_exp=known_median[["excitatory_percentage", "inhibitory_percentage"]],
)
