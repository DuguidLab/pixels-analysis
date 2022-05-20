# This will plot a figure aligned to LED off with the following features:
# A series of indicators displaying when LED came on across trials for each unit, plotted as a histogram
# The confidence interval for the spike rate.

# First shall import the required packages
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


##Now import the main experimental data. Here all good units from cortex##
unit = myexp.select_units(group="good", uncurated=False, max_depth=1500)

# Now must align these trials to LED off for the main spike rate data
hits_off = myexp.align_trials(
    ActionLabels.correct,
    Events.led_off,
    "spike_rate",
    duration=6,  # May need to increase this depending on how far back the LED came on
    units=unit,
)

##Now will obtain the data containing time at which the LED turned on and duration##

# I shall iterate through the trial taking the difference between the event on (LED_on) and event off (LED_off)
# This will give a duration of events and allow plotting
# First iterate through the experimental data, both sessions and get the action labels and events
# Importantly, this will be structured as a function to allow me to run this easily in future


##The following function will return the timepoints for the requested event type for a given experiment
#
def event_times(event, myexp):
    """

    This function will give the timepoints for the specified event across experimental sessions

    event: the specific event to search for, must be input within quotes ("")

    myexp: the experiment defined in base.py

    NB: Setting the event to LED_off when the action label is correct (whose start point therefore defines LED_on as zero) will return the DURATION of the event!
    """
    times = []  # Give an empty list to store times

    for ses in myexp:
        sessiontimepoints = (
            []
        )  # Give an empty list to store this sessions times in before appending to the master list
        als = ses.get_action_labels()  # Get the action labels for this session

        for rec_num in range(
            len(als)
        ):  # This will run through all recording numbers in the above action label data
            actions = als[rec_num][:, 0]
            events = als[rec_num][:, 1]

            # Call all trials where the mouse was correct to search
            start = np.where(np.bitwise_and(actions, ActionLabels.correct))[0]

            for trial in start:
                # Now iterate through these correct trials and return all times for selected event
                event_time = np.where(
                    np.bitwise_and(
                        events[trial : trial + 10000], getattr(Events, event)
                    )
                )[0]
                sessiontimepoints.append(event_time[0])

        times.append(sessiontimepoints)

    return times


##Now let us define the event times, and durations for the experiment
led_duration = event_times("led_off", myexp)


# Combine these as a dictionary, now have all on times across sessions. Each representing an individual trial
rugdata = {ses.name: pd.DataFrame(led_duration[s]) for s, ses in enumerate(myexp)}

rugdata = pd.concat(rugdata, axis=1)
rugdata = (
    -rugdata / 1000
)  # Converts the positive values to a negative second, to signify the duration before the LED off

##Now will plot the base information, combined with a rugplot addition
# To do so, must edit the _plot function, imported from rugplotbase
for s, session in enumerate(myexp):
    name = session.name

    per_unit_spike_rate(hits_off[s], rugdata[name], ci="sd")
    plt.suptitle(
        f"session {name} - per-unit across trial firing rate, aligned to LED off"
    )

plt.show()
