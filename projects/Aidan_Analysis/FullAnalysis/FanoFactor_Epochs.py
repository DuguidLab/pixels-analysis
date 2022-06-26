# This file will calculate the fano factor relative to different alignment points.
# Then compare the changes observed.
# First import required packages
from argparse import Action
import math

from matplotlib.pyplot import axvline, xlim
from base import *
import numpy as np
import pandas as pd
import matplotlib as plt
from pixtools.utils import Subplots2D
from rasterbase import *
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from pixtools.utils import Subplots2D
from pixels import ioutils
import statsmodels.api as sm
from statsmodels.formula.api import ols

from functions import (
    event_times,
    per_trial_raster,
    per_trial_binning,
    per_unit_spike_rate,
    bin_data_average,
    cross_trial_FF,
    fano_fac_sig,
)

# Import previously calculated FF values
population_FF = ioutils.read_hdf5(
    "/data/aidan/per_unit_FF_FF_Calculation_mean_matched_True_rawTrue.h5"
)
population_FF = population_FF.melt()
population_FF.rename(columns={"variable": "bin", "value": "FF"}, inplace=True)
# Calculate any significant changes in FF across time: using the SE produced from the model
