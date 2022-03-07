#This script draws a confidence interval comparison between a baseline and some event. This is calculated as baseline - average firing rate 
#Will use a modified version of a preexisting function in base.py (align_spike_rate_CI)
#Import required packages
from argparse import Action
import sys
from tokenize import group
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from pandas import melt
import seaborn as sns
from textwrap import wrap

sys.path.insert(0, "/home/s1735718/PixelsAnalysis/pixels") #use  the local copy of base.py
from base import * 

#First select units to analyse
units = myexp.select_units(
    group="good",
    max_depth=3500,
    name="unit"
)
#Now let us run the align spike rate CI and save this as CIs

CIs = myexp.get_aligned_spike_rate_CI(
    label=ActionLabels.correct_left | ActionLabels.correct_right,
    event=Events.led_off,
    start=-0.400, #Start the analysis 400ms before LED turns off
    step=0.400, #Make this analysis in one step, not looking after or before
    end=0.000, #Finish the analysis for each unit when the grasp is made

    bl_label=ActionLabels.correct_left | ActionLabels.correct_right,
    bl_event=Events.led_on, #The baseline will examine the ITI (i.e., time before the trial began)
    bl_start=-5.000,
    bl_end=0.000,

    ss=20, #The size of each sample to take
    CI=95, #Confidence interval to analyse to 
    bs=10000, #The number of times to take a pseudorandom sample for bootstrapping
    units=units
)

#Also create a dataset containing only values that do not straddle zero. 
#Could check if first and last percentile are different from zero 
#cis[0] > 0 or cis[-1] < 0 - if true then suggests significance!
#Will be easier if this is in wide form, can simply take it by column

#First will iterate through the sessions for CI
#Remember the indexes are as follows:
#Session, Unit, rec_num (always zero), percentile (2.5, 50, or 97.5)

def significance_extraction(CI):
    """
    This function takes the output of the get_aligned_spike_rate_CI function and extracts any significant values, returning a dataframe in the same format. 

    CI: The dataframe created by the CI calculation previously mentioned

    """
    
    sig = []
    keys=[]
    rec_num = 0

    #This loop iterates through each column, storing the data as un, and the location as s
    for s, unit in CI.items():
        #Now iterate through each recording, and unit
        #Take any significant values and append them to lists.
        if unit.loc[2.5] > 0 or unit.loc[97.5] < 0:
            sig.append(unit) #Append the percentile information for this column to a list
            keys.append(s) #append the information containing the point at which the iteration currently stands


    #Now convert this list to a dataframe, using the information stored in the keys list to index it
    sigs = pd.concat(
        sig, axis = 1, copy = False,
        keys=keys,
        names=["session", "unit", "rec_num"]
    )
    
    return sigs

sigs = significance_extraction(CIs)

#Now we have successfully derived confidence intervals, we may plot these as a scatter for each unit, with a line denoting the critical point
CIs_l_sig=sigs.reset_index().melt("percentile").sort_values("value", ascending=False) #Convert the dataframe to long form, then order by value increasing
CIs_l_sig = CIs_l_sig.reset_index(drop=True)

for s, session in enumerate(myexp):
    name = session.name

    p = sns.stripplot(
        x="unit", y = "value", data = CIs_l_sig.loc[(CIs_l_sig.session == s)], hue = "percentile"
        )
    p.set_xlabel("Unit")
    p.set_ylabel("Confidence Interval")
    #p.set(xticklabels=[])
    p.axhline(0)
    plt.suptitle("\n".join(wrap(f"Confidence Intervals By Unit - Grasp vs. Baseline - Session {name}")))

    plt.show()


#Can also make a pie chart detailing the proportions of the whole collection of units that were signifcant
#Can simply compare the lengths of the unique values in the unit columns of the dataframes
#Why are these exactly half? Run another experimental session to check!
 
for s, session in enumerate(myexp):
    prop = []
    ses_data = CIs[s]
    sig_data = sigs[s].reset_index().melt("percentile").sort_values("value", ascending=False)
    sig_data = sig_data.reset_index(drop=True)

    #Get a longform dataframe of all CIs calcluated across units
    CIs_long = ses_data.reset_index().melt("percentile").sort_values("value", ascending=False)
    CIs_long = ses_data.reset_index(drop=True)

    #Now calculate the proportion of significant to nonsignificant units
    #Remember due to the different methods of storage these functions differ
    sig_count = len(sig_data.unit.unique())
    nonsig_count = len(ses_data.columns) - sig_count

    prop.append(sig_count)
    prop.append(nonsig_count)
    
    #Now plot a pie chart
    plt.pie(prop, labels = ["Significant Units", "Non-significant Units"])
    plt.show()





