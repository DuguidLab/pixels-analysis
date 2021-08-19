'''
Compare variances and skewness of two distributions
'''
from scipy import stats
import statistics as sts
import numpy as np
import pandas as pd

def compare_var(naive, expert, alpha=0.05):
    """
    F-test to compare variances of two distributions.

    parameters
    ====
    naive: list of values from naive mice.

    expert: list of values from expert mice.

    alpha: significance

    return
    ====
    f: float, F-test statistic

    p: float, p-value of F-test statistic
    """

    naive_size, expert_size = len(naive), len(expert)
    naive_v, expert_v = sts.variance(naive), sts.variance(expert)

    f = expert_v / naive_v
    df_naive = naive_size - 1
    df_expert = expert_size - 1
    p = stats.f.sf(f, df_expert, df_naive)

    print('\nF =', f, 'p =', p) 
    if p < alpha:
        print('\nvariances are significantly different.')
    else:
        print('\nvariances are not significantly different.')

    return f, p


def compare_skew(naive, expert, names=None):
    '''
    Compare the skewness of two distributions.

    parameters
    ====
    
    naive: list of values from naive mice.

    expert: list of values from expert mice.

    names: list of strings

    return
    ====

    naive_skew_ci: df, 95% confidence intervals of the naive population skewness

    expert_skew_ci: df, 95% confidence intervals of the expert population skewness
    '''

    naive_size, expert_size = len(naive), len(expert)
    naive_v, expert_v = sts.variance(naive), sts.variance(expert)
    df_naive = naive_size - 1
    df_expert = expert_size - 1

    # Fisher-Pearson coefficient of skewness
    naive_skew = stats.skew(naive, bias=False)
    print(f'\n{names[0]} skewness coeffcient is', naive_skew)
    if abs(naive_skew) > 1:
        print(f'\nthis {names[0]} sample is highly skewed.')
    elif 0.5 < abs(naive_skew) < 1:
        print(f'\nthis {names[0]} sample is moderately skewed.')
    elif -0.5 < naive_skew < 0.5:
        print(f'\nthis {names[0]} sample is approximately symmetric.')

    expert_skew = stats.skew(expert, bias=False)
    print(f'\n{names[1]} skewness coeffcient is', expert_skew)
    if abs(expert_skew) > 1:
        print(f'\nthis {names[1]} sample is highly skewed.')
    elif 0.5 < abs(expert_skew) < 1:
        print(f'\nthis {names[1]} sample is moderately skewed.')
    elif -0.5 < expert_skew < 0.5:
        print(f'\nthis {names[1]} sample is approximately symmetric.')

    # standard error of skewness
    ses_naive = ((6 * naive_size * df_naive)/((naive_size - 2) * (naive_size + 1) * (naive_size + 3))) ** 0.5
    ses_expert = ((6 * expert_size * df_expert)/((expert_size - 2) * (expert_size + 1) * (expert_size + 3))) ** 0.5

    # inferring skewness of population
    z_naive = naive_skew / ses_naive
    z_expert = expert_skew / ses_expert

    if z_naive > 2:
        print(f'\nthe {names[0]} population is likely skewed positively (though I don’t know by how much).')
    if z_expert > 2:
        print(f'\nthe {names[1]} population is likely skewed positively (though I don’t know by how much).')

    # 95% confidence intervals of skewness
    skew_ci = []
    percentile = [2.5, 50, 97.5]
    skew_ci.append([naive_skew - 2 * ses_naive, naive_skew, naive_skew + 2 * ses_naive])
    skew_ci.append([expert_skew - 2 * ses_expert, expert_skew, expert_skew + 2 * ses_expert])
    skew_ci_df = pd.DataFrame(skew_ci, index=names, columns=percentile).T
    print(f'\n95% CI of population skewness:\n', skew_ci_df)

    if skew_ci_df[names[1]][2.5] > skew_ci_df[names[0]][97.5]:
        print(f'\nthe {names[1]} population is likely to be more positively skewed than {names[0]} population')

    return skew_ci_df
