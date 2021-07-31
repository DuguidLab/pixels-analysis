# Calculate coding accuracies for each unit and save to cache

from pixels import Experiment
from pixels.behaviours.reach import ActionLabels, Events, VisualOnly
from pixtools.naive_bayes.unit_decoding_accuracy import gen_unit_decoding_accuracies
from naive_ipsi_contra_spike_rate import *

# for M2, left is ipsi, right is contra.
rec_num = 0


for s, session in enumerate(exp):
    gen_unit_decoding_accuracies(
        session, stim_left[s][rec_num], stim_right[s][rec_num], "direction", force=True
    )
