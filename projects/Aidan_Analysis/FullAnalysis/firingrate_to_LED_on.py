from argparse import Action
import sys
from cv2 import bitwise_and
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from base import *
from rugplotbase import *
from matplotlib.pyplot import figure
from distutils.command.sdist import sdist
from textwrap import wrap

sys.path.append("/home/s1735718/PixelsAnalysis/pixels-analysis") 
from pixtools import spike_rate
from pixtools import utils


##Now import the main experimental data. Here all good units from cortex##
unit = myexp.select_units(
    group="good",
    uncurated=False,
    max_depth=1500,
    name="cortex"
)

#Now must align these trials to LED on for the main spike rate data
hits_off = myexp.align_trials(
    ActionLabels.correct,
    Events.led_on,
    "spike_rate",
    duration=8,
    units=unit
)

#Now plot average firing rate per unit, across trials. Will only examine the 4 seconds preceeding 
ci = "sd"

for s, session in enumerate(myexp):

    fig = plt.figure()
    name = session.name
    spike_rate.per_unit_spike_rate(hits_off[s], ci=ci)
    fig.suptitle(
        f"session {name} - per-unit firing rate 4s before LED on (Trial Beginning)"
    )
    plt.gcf().set_size_inches(20, 20)
    utils.save(f"/home/s1735718/Figures/{myexp[s].name}_spikerate_LED_on", nosize=True)
    #plt.show()


