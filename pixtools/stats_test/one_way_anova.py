'''
One-way ANOVA using ordinary least suqares (OLS) model.
'''
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.stats.multicomp as multi
from scipy import stats

import numpy as np
import pandas as pd

from pixels import ioutils

def ow_anova(df, num=None, cat=None):
    '''
    Comparing means from multiple (sub)groups by fitting to ordinary least
    squares (OLS) model, and do one-way ANOVA.

    If ANOVA is significant, do post hoc multiple comparison. Default test is Tukey HSD.

    parameters:
    ====
    df: dataframe.

    num: string, name of numerical (interval) data.

    cat: string, name of categories.

    '''
    # melt df, put categorical data in one colum, and numerical data in another.
    df = df.melt(value_name=num, var_name=cat)

    # fit OLS
    df_ols = ols(f'{num}~C({cat})', data=df).fit()
    print('\nOLS model summary: \n', df_ols.summary())

    df_anova = anova_lm(df_ols)
    print(df_anova)

    if df_anova['PR(>F)'][0] < 0.05:
        print('\none-way ANOVA is significant, doing post-hoc tests:')

        post_hoc = multi.MultiComparison(df[num], df[cat]).tukeyhsd()

        print('\nPost hoc tests results: \n', post_hoc.summary())
