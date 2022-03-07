#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:12:43 2022

@author: s1735718
"""
# Import experimental instance
from base import *

myexp.set_cache("overwrite")

# This selects the specific neuronal units will be included in the plots
units = myexp.select_units(
    group="good",  # Gives quality of units, here good only
    max_depth=1500,  # Give us units above a depth of 1500um
    name="cortex",  # The name to cache the info under
    uncurated = True #If dealing with pre-phy data be sure to specify "uncurated = TRUE" in the select_units function
)

assert units[0]  #Are there any units in this list? If not then this will raise an error!


# Plotting all behavioural data channels for session 1, trial 3
hits = myexp.align_trials(
    ActionLabels.correct,  # This selects which trials we want
    Events.led_on,  # This selects what event we want them aligned to
    "spike_rate",  # And this selects what kind of data we want
    duration=4,  # Gives us two seconds on either side of the event
    units=units,
)

#First plot 6 empty figures
plt.figure()
fig, axes = plt.subplots(6, 1, sharex=True)
channels = hits.columns.get_level_values("unit").unique() #Give all the unique values in the "unit" column of hits, where hits gives us every trial with a successful reach
trials = hits.columns.get_level_values("trial").unique() #Give all the unique values in the "trial" column of hits, i.e. the trial n0 where the mouse successfully reached 
trial = trials[1] #Set trial to be the nth item in the list of trials, whatever number this may take!
session = 0 #The first recording session will be analysed

# #Now iterate over the data, channel by channel and give all data concerning times hits occurred
# for i in range(6):
#     chan_name = channels[i]
#     sns.lineplot(
#         data=hits[session][chan_name][trial], estimator=None, style=None, ax=axes[i]
#     )

# plt.show() #Plot this information


# # Plotting spike data from session 1, trial 8, units 151 to 161
# hits = myexp.align_trials(ActionLabels.correct, Events.led_on, "spikes")

plt.figure()
fig, axes = plt.subplots(10, 1, sharex=True)
trial = trials[7]
session = 0

for i in range(10):
    sns.lineplot(data=hits[151 + i][trial], estimator=None, style=None, ax=axes[i])
plt.show()
