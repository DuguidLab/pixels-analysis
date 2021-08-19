'''
One-way ANOVA using ordinary least suqares (OLS) model.
'''
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.stats.multicomp as multi
from scipy import stats

import numpy as np
import pandas as pd
import scikit_posthocs as sp

from pixels import ioutils

def ow_anova(df, num=None, cat=None, parametric=True):
    '''
    Comparing means from multiple (sub)groups by fitting to ordinary least
    squares (OLS) model, and do one-way ANOVA.

    If ANOVA is significant, do post hoc multiple comparison. Default test is Tukey HSD.
    
    If Kruskal-Wilk is significant, do post hoc multiple comparison. Default test is Conover.

    parameters:
    ====
    df: dataframe.

    num: string, name of numerical (interval) data.

    cat: string, name of categories.

    parametric: bool, decide whether to use parametric (ANOVA) or
    non-parametric eqivalent (Kruskal-Wilk) based on Durbin-Watson test &
    normality check.
        Default is True.

    '''
    # assumptions check
    # normality, Shapiro-Wilk (p>0.05?)
    normality_check = []
    for i in range(len(df.columns)):
        W,W_p = stats.shapiro(df[df.columns[i]])
        normality_check.append((W, W_p))
    normality = pd.DataFrame(normality_check, columns=['W', 'p'])
    print('normality check:\n', normality)

    # melt df, put categorical data in one colum, and numerical data in another.
    df_melt = df.melt(value_name=num, var_name=cat)

    if parametric:
        # fit OLS
        df_ols = ols(f'{num}~C({cat})', data=df_melt).fit()
        print('\nOLS model summary: \n', df_ols.summary())
        '''
        independence, Durbin-Watson (independent data tend to have DW around
        2. A positive correlation makes DW smaller and negative correlation
        makes it bigger. 1.5<DW<2.5?)
        '''
        df_anova = anova_lm(df_ols)
        print(df_anova)

        if df_anova['PR(>F)'][0] < 0.05:
            print('\none-way ANOVA is significant, doing post-hoc tests:')

            # post hoc tukey HSD
            post_hoc = multi.MultiComparison(df_melt[num], df_melt[cat]).tukeyhsd()
            print('\nPost hoc tests results: \n', post_hoc.summary())

    else:
        H, H_p = stats.kruskal(*[c for cat, c in df_melt.groupby(cat)[num]])
        print('Kruskal-Wilk test statistic =', H, 'p =', H_p)

        if H_p < 0.05:
            print('\nKruskal-Wilk is significant, doing post-hoc tests:')
            
            # post hoc Conover
            post_hoc = sp.posthoc_conover(df_melt, val_col=num, group_col=cat, p_adjust = 'holm')
            print('\nPost hoc tests results: \n', post_hoc)
