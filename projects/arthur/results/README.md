This folder contains results of npx analysis.

usage: ioutils.read_hdf5(path)
=====

*group: naive/expert*

- naive: HFR20, HFR22, HFR23 

- expert: HFR25, HFR29 (right ppc recording only)

*area: m2/ppc*

*duration: duration of the event alignment window, 1 or 2 seconds*

- group_area_sig_corr_units_count_stim-side_duration: number of significantly correlated units from the other area that the **area** units has from the other area

- group_area_resps_sig_corr_units_count: number of significantly correlated units from the other area that the **area** units has from the other area **RESPONSIVE UNITS ONLY**
e.g., if area=m2, how many ppc units are m2 units significantly correlated to?

- area_sig_corr_units_stats: descriptive statistics of the above dataset.

- group_area_resps_units_bias_groups: **ONLY IN NAIVE** number of responsive units in bias groups (ipsi bias, contra bias, no bias, or respond in opposite directions) 

- group_area_resps_units_count: total number of units, number of responsive units, proportion of responsive units. 

**NOTE**: in expert, 'right ppc' is from HFR29 who only had one recording in right ppc. consider to append it to expert_ppc. 

- group_area_resps_units: unit ids of all respsonsive units from the area.

- group_max_cc_duration: the maximum (positive & negative) correlation coefficient in each condition and its m2-ppc id.

- group_resps_max_cc_duration: the maximum (positive & negative) correlation coefficient in each condition and its m2-ppc id, **RESPONSIVE UNITS ONLY**
