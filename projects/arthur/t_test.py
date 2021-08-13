'''
indenpendent t-test and effect size (cohen's d) test.
'''
from scipy import stats
import statistics as sts
import numpy as np

def cohens_d(naive, expert):
    """
    calculate cohen's d, the effect size, of an independent t-test.
    ===
    naive: list of values from naive mice, usually is proportion.

    expert: list of values from expert mice, usually is proportion.
    """
    naive_size, expert_size = len(naive), len(expert)
    naive_v, expert_v = sts.variance(naive), sts.variance(expert)
    s = (((naive_size - 1) * naive_v + (expert_size - 1) * expert_v) / (naive_size + expert_size - 2)) ** 0.5
    naive_mean, expert_mean = sts.mean(naive), sts.mean(expert)
    d = abs(expert_mean - naive_mean) / s

    if d > 0.8:
        print('large effect size :)')
    elif d < 0.2:
        print('small effect size :(')
    else:
        print('medium effect size.')

    return d

def t_test(naive, expert):
    """
    independent t-test between naive & expert mice.
    ===
    naive: list of values from naive mice, usually is proportion.

    expert: list of values from expert mice, usually is proportion.
    """
    t, p = stats.ttest_ind(naive, expert)
    
    return t, p

test append lol