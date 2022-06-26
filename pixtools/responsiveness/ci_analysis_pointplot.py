def significance_extraction(CI):
    """
    This function takes the output of the get_aligned_spike_rate_CI method under the myexp class and extracts any significant values, returning a dataframe in the same format. 

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

def percentile_plot(CIs, sig_CIs, exp, sig_only = False, dir_ascending = False):
    """

    This function takes the CI data and significant values and plots them relative to zero. 
    May specify if percentiles should be plotted in ascending or descending order. 

    CIs: The output of the get_aligned_spike_rate_CI function, i.e., bootstrapped confidence intervals for spike rates relative to two points.

    sig_CIs: The output of the significance_extraction function, i.e., the units from the bootstrapping analysis whose confidence intervals do not straddle zero
    
    exp: The experimental session to analyse, defined in base.py

    sig_only: Whether to plot only the significant values obtained from the bootstrapping analysis (True/False)

    dir_ascending: Whether to plot the values in ascending order (True/False)
    
    """
    #First sort the data into long form for the full dataset, by percentile
    CIs_long = CIs.reset_index().melt("percentile").sort_values("value", ascending= dir_ascending)
    CIs_long = CIs_long.reset_index()
    CIs_long["index"] = pd.Series(range(0, CIs_long.shape[0]))#reset the index column to allow ordered plotting

    #Now select if we want only significant values plotted, else raise an error. 
    if sig_only is True:
        CIs_long_sig = sig_CIs.reset_index().melt("percentile").sort_values("value", ascending=dir_ascending)
        CIs_long_sig = CIs_long_sig.reset_index()
        CIs_long_sig["index"] = pd.Series(range(0, CIs_long_sig.shape[0]))
        
        data = CIs_long_sig
    
    elif sig_only is False:
        data = CIs_long
    
    else:
        raise TypeError("Sig_only argument must be a boolean operator (True/False)")

    #Plot this data for the experimental sessions as a pointplot. 
    for s, session in enumerate(exp):
        name = session.name

        p = sns.pointplot(
        x="unit", y = "value", data = data.loc[(data.session == s)],
        order = data.loc[(data.session == s)]["unit"].unique(), join = False, legend = None) #Plots in the order of the units as previously set, uses unique values to prevent double plotting
        
        p.set_xlabel("Unit")
        p.set_ylabel("Confidence Interval")
        p.set(xticklabels=[])
        p.axhline(0)
        plt.suptitle("\n".join(wrap(f"Confidence Intervals By Unit - Grasp vs. Baseline - Session {name}"))) #Wraps the title of the plot to fit on the page.

        plt.show()