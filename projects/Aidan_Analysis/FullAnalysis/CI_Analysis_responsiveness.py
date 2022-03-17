#This file will produce graphs detailing the relative responsiveness of units. Will have three scatters:
#Did not significantly change (unresponsive), increased in activity, decreased in activity. 
#Can import data from CI Analysis

import sys
from base import *
from CI_Analysis import significance_extraction
from CI_Analysis import percentile_plot

sys.path.append("/home/s1735718/PixelsAnalysis/pixels-analysis")
from pixtools.utils import Subplots2D
sys.path.insert(0, "/home/s1735718/PixelsAnalysis/pixels") #use  the local copy of base.py



#First select the units that shall be analysed
units = myexp.select_units(
    group="good",
    max_depth=3500,
    name="unit"
)

#Then align spike rates to trial, and gennerate confidence intervals
#Will once again compare peak reach time to a 4s ITI period acting as baseline
#Think this is still trying to use the old version, update from git?

CIs = myexp.get_aligned_spike_rate_CI(
    label=ActionLabels.correct_left | ActionLabels.correct_right,
    event=Events.led_off,
    start=-0.400, #Start the analysis 400ms before LED turns off
    step=0.400, #Make this analysis in one step, not looking after or before
    end=0.000, #Finish the analysis for each unit when the grasp is made

    bl_label=ActionLabels.correct_left | ActionLabels.correct_right,
    bl_event=Events.led_on, #The baseline will examine the ITI (i.e., time before the trial began)
    bl_start=-4.000, #Will catch every trial's ITI while avoiding the tail end of the previous trial 
    bl_end=0.000,

    ss=20, #The size of each sample to take
    CI=95, #Confidence interval to analyse to 
    bs=10000, #The number of times to take a pseudorandom sample for bootstrapping
    units=units
)

