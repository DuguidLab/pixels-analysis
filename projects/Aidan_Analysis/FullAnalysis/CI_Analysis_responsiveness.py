#This file will produce graphs detailing the relative responsiveness of units. Will have three scatters:
#Did not significantly change (unresponsive), increased in activity, decreased in activity. 
#Can import data from CI Analysis

from matplotlib.pyplot import title, ylim
import numpy as np
import pandas as pd

import sys
from base import *
from CI_Analysis import significance_extraction
from CI_Analysis import percentile_plot

sys.path.append("/home/s1735718/PixelsAnalysis/pixels-analysis")
sys.path.insert(0, "/home/s1735718/PixelsAnalysis/pixels/pixels")
from pixtools.utils import Subplots2D #use the local copy of base.py
from pixtools import utils


#First select the units that shall be analysed
units = myexp.select_units(
    group="good",
    max_depth=3500,
    name="unit"
)

#Then align spike rates to trial, and gennerate confidence intervals
#Will once again compare peak reach time to a 4s ITI period acting as baseline
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

#Now take this information and pass it to the significance extraction function
sigs = significance_extraction(CIs)

#Split this data into upregulated and downregulated units
#Define a function to split units by responsiveness
def responsiveness(CI, sigs_only = True):

    """
    This function takes the confidence intervals computed by the aligned 
    spike rate CI method and returns the responsiveness type (Up, Down, No Sig Change)

    If sigs_only is true (default) then this function will not return those values that did not significantly change from zero

    CI: The data produced by the aligned spike rate CI method of myexp

    sigs_only: Whether to return no change values

    """

    #first begin by extracting significant values
    sigs = significance_extraction(CI)

    #Then create empty lists to store the data
    units = []
    keys = []
    rec_num = 0

    #If we only want the significant units:
    if sigs_only == True:
        #Check if the 2.5 percentile is greater than or equal to zero (i.e., upregulated is true)
        #Append this as a new row
        upregulated = (sigs.loc[2.5] >= 0).rename(index={2.5: "change"})
        sigs.loc["upregulated"] = upregulated #True or False that unit was upregulated

        return sigs

    #Or if we want every unit regardless of significance:    
    elif sigs_only == False:
        #Check the entire dataframe rather than iterating, using a boolean operator

        CI.loc["change"] = "nochange" #Add a row of entirely no change values (default)
        
        upregulated = (CI.loc[2.5] >= 0).rename(index={2.5: "change"}) #Find all upregulated values
        downregulated = (CI.loc[97.5] <= 0).rename(index={2.5: "change"}) #And the same for downregulated values
        
        CI.loc["change"][upregulated] = "up" #Replace these values with up
        CI.loc["change"][downregulated] = "down"
        print(CI)
        return CI


df = responsiveness(CIs, sigs_only=False)

#Now transpose this to long form information and plot this by depth
#To get depth for a unit, use get cluster info, here cluster_id is the unit, and depth is stated
#Get cluster_id == unit and extract depth. 
info = myexp.get_cluster_info()
depths = []
keys = []
units = []
skipped = []


#Now iterate through the list, finding the depth of each unit

for s, unit in df.items():

    ses_info = info[s[0]]

    if len(ses_info["depth"].loc[ses_info["cluster_id"] == s[1]]) == 0:
        skipped.append(s[1])
        continue #This line checks if the unit is absent from cluster_id and skips it if it is, will also append this to a list for checking later

    units.append(unit)
    key = list(s)
    depths = ses_info["depth"].loc[ses_info["cluster_id"] == s[1]].item()
    key.append(depths)
    keys.append(tuple(key))


df2 = pd.concat(
    units, axis = 1, copy = False,
    keys = keys, names = ["session", "unit", "rec_num", "depth"]
)


#Once the units are known alongside their depths, can plot the relative proportion of response types
#Will use a bar plot for this
#Iterate over session in df2, plotting the chart for each

fig, axs = plt.subplots(2)
plt.subplots_adjust(top = 1, hspace=0.5)
for s, session in enumerate(df2):
    name = myexp[session[0]].name
    sns.countplot(
        x = df2[session[0]].xs("change"), data = df2[session[0]], orient='vertical',
        palette="pastel", ax = axs[session[0]]
    ).set_title(f"{name} Count of Unit Responsiveness")

#Now save this figure including all sessions used. 
names = "_".join(s.name for s in myexp)
utils.save(f"/home/s1735718/Figures/{names}_count_of_unit_responsiveness", nosize=True)