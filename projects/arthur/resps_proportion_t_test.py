'''
list of responsive unit proportion for naive & trained mice, with 250ms ci bin size
'''
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pixtools import stats_test

# 200ms bin
#naive_m2 = [0.07189, 0, 0.1, 0.02041, 0.03947, 0.01299]
#expert_m2 = [0.12941, 0.08547]
#d_m2 = cohens_d(naive_m2, expert_m2)
#t_m2, p_m2 = t_test(naive_m2, expert_m2)
#
#naive_ppc = [0.03158, 0.15517, 0.13462, 0.13402, 0.19328, 0.15108]
#expert_ppc = [0.12844,0.07317, 0.30769]
#d_ppc = cohens_d(naive_ppc, expert_ppc)
#t_ppc, p_ppc = t_test(naive_ppc, expert_ppc)

# 200ms rolling bin
naive_m2 = [0.07784, 0.05, 0.15, 0.02041, 0.05263, 0.01299]
expert_m2 = [0.12941, 0.08547]
d_m2 = stats_test.cohens_d(naive_m2, expert_m2)
t_m2, p_m2 = stats_test.t_test(naive_m2, expert_m2)

naive_ppc = [0.04211, 0.17241, 0.17308, 0.16495, 0.21008, 0.16547]
expert_ppc = [0.12844,0.07317, 0.30769]
d_ppc = stats_test.cohens_d(naive_ppc, expert_ppc)
t_ppc, p_ppc = stats_test.t_test(naive_ppc, expert_ppc)
print('naive m2 mean: ', sts.mean(naive_m2), 'expert m2 mean: ', sts.mean(expert_m2))

print('naive-expert 2 t-test result: t = ', t_m2, 'p = ', p_m2)

print('naive-expert ppc t-test result: t = ', t_ppc, 'p = ', p_ppc)

print('effect size: ', 'm2 effect size: ', d_m2, 'ppc effect size: ', d_ppc) 

# within naive & expert, difference between ppc & m2
naive_t,naive_p = stats_test.t_test(naive_ppc, naive_m2)
naive_d = stats_test.cohens_d(naive_ppc, naive_m2)
print('naive ppc-m2: ', 't = ', naive_t, 'p = ', naive_p, 'd = ', naive_d) 

expert_t,expert_p = stats_test.t_test(expert_ppc, expert_m2)
expert_d = stats_test.cohens_d(expert_ppc, expert_m2)
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
