'''
list of responsive unit proportion for naive & trained mice, with 250ms ci bin size
'''
from scipy import stats
import statistics as sts
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#def sig_test(naive_p, expert_p, bin_size=200, std, cohens_d):
    
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


#
#
# 200ms bin
naive_m2 = [0.07189, 0, 0.1, 0.02041, 0.03947, 0.01299]
expert_m2 = [0.12941, 0.08547]
d_m2 = cohens_d(naive_m2, expert_m2)
t_m2, p_m2 = t_test(naive_m2, expert_m2)

naive_ppc = [0.03158, 0.15517, 0.13462, 0.13402, 0.19328, 0.15108]
expert_ppc = [0.12844,0.07317, 0.30769]
d_ppc = cohens_d(naive_ppc, expert_ppc)
t_ppc, p_ppc = t_test(naive_ppc, expert_ppc)

print(sts.mean(naive_m2), sts.mean(expert_m2))

print('m2 t-test result: t = ', t_m2, 'p = ', p_m2)

print('ppc t-test result: t = ', t_ppc, 'p = ', p_ppc)

print('effect size: ', 'm2 effect size: ', d_m2, 'ppc effect size: ', d_ppc) 

# within naive & expert, difference between ppc & m2
naive_t,naive_p = t_test(naive_ppc, naive_m2)
naive_d = cohens_d(naive_ppc, naive_m2)
print('naive ppc-m2: ', 't = ', naive_t, 'p = ', naive_p, 'd = ', naive_d) 

expert_t,expert_p = t_test(expert_ppc, expert_m2)
expert_d = cohens_d(expert_ppc, expert_m2)
print('expert ppc-m2: ', 't = ', expert_t, 'p = ', expert_p, 'd = ', expert_d) 

#print(effect_size_check(d_m2), effect_size_check(d_ppc))
'''
list of responsive unit proportion for naive & trained mice, with 250ms ci bin size
'''
#naive_m2 = [0.06536, 0, 0.15, 0.01020, 0.03509, 0]
#expert_m2 = [0.12941, 0.06838]
#stats.stdev(naive_m2)
#stats.stdev(expert_m2)
#t, p = stats.ttest_ind(naive_m2, expert_m2)
#print('variances are: ', stats.stdev(naive_m2), stats.stdev(expert_m2))
#print('independent t-test between naive & trained m2 responsiveness (250ms bin): ', t, p)
#sns.boxplot(data=naive_ppc)
#sns.swarmplot(data=naive_ppc)
#plt.show()
