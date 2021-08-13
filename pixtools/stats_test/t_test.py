'''
indenpendent t-test and effect size (cohen's d) test.
'''
from scipy import stats
import statistics as sts
import numpy as np

def get_t_p_d(naive, expert, equal_var=True):
    """
    independent t-test between naive & expert mice, and cohen's d, the effect size, of an independent t-test.
    ===
    naive: list of values from naive mice, usually is proportion.

    expert: list of values from expert mice, usually is proportion.

    equal_var: assumption that two populations have the same variances. Default is True.
        if 1/2 < (s1/s2) < 2, equal_var = True;
        else: equal_var = False, will perform Welch's t-test.
    """
    if equal_var == True:
        t, p = stats.ttest_ind(naive, expert)
    else:
        t, p = stats.ttest_ind(naive, expert, equal_var=False)
    
    naive_size, expert_size = len(naive), len(expert)
    naive_v, expert_v = sts.variance(naive), sts.variance(expert)
    s = (((naive_size - 1) * naive_v + (expert_size - 1) * expert_v) / (naive_size + expert_size - 2)) ** 0.5
    naive_mean, expert_mean = sts.mean(naive), sts.mean(expert)
    d = abs(expert_mean - naive_mean) / s

    print('\nt = ', t, 'p = ', p, 'd = ', d) 
    if d > 0.8:
        print('large effect size :)')
    elif d < 0.2:
        print('small effect size :(')
    elif 0.2 <= d <= 0.8:
        print('medium effect size.')

    return t, p, d




