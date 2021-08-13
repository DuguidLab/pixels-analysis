'''
test responsive interneuron/pyramidals proportion difference between naive & trained mice.
'''

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pixtools import stats_test

#naive:
py_proportion_all_PPC = [0.946236559139785, 0.8392857142857143, 0.7708333333333334, 0.8695652173913043, 0.8608695652173913, 0.7703703703703704]
py_proportion_resps_PPC = [0.6666666666666666, 0.7777777777777778, 0.625, 0.625, 0.8636363636363636, 0.6363636363636364]
py_proportion_all_M2 = [0.9266666666666666, 0.8125, 0.9, 0.8877551020408163, 0.9111111111111111, 0.9342105263157895]
py_proportion_resps_M2 = [0.5833333333333334, 1.0, 1.0, 0.6666666666666666, 0.9090909090909091, 1.0]

intn_proportion_all_M2 = [0.07333333333333336, 0.1875, 0.09999999999999998, 0.11224489795918369, 0.0888888888888889, 0.06578947368421051]
intn_proportion_resps_M2 = [0.41666666666666663, 0.0, 0.0, 0.33333333333333337, 0.09090909090909094, 0.0]
intn_proportion_all_PPC = [0.053763440860215006, 0.1607142857142857, 0.22916666666666663, 0.13043478260869568, 0.13913043478260867, 0.22962962962962963]
intn_proportion_resps_PPC = [0.33333333333333337, 0.2222222222222222, 0.375, 0.375, 0.13636363636363635, 0.36363636363636365]

#trained
py_proportion_all_ppc = [0.75, 0.8181818181818182, 1.0]
py_proportion_resps_ppc = [0.7857142857142857, 1.0, 1.0]
py_proportion_all_m2 =  [0.8452380952380952, 0.7807017543859649]
py_proportion_resps_m2 =  [0.7272727272727273, 0.8]

intn_proportion_all_M2 = [0.15476190476190477, 0.2192982456140351]
intn_proportion_resps_M2 = [0.2727272727272727, 0.19999999999999996]
intn_proportion_all_PPC = [0.25, 0.18181818181818177]
intn_proportion_resps_PPC = [0.2142857142857143, 0.0]

'''
now can look at how many py & int in each responsive bias group. But too much work for now, drop it :))
'''

assert False
#print('naive ppc-m2: ', 't = ', naive_t, 'p = ', naive_p, 'd = ', naive_d) 
#print('expert ppc-m2: ', 't = ', expert_t, 'p = ', expert_p, 'd = ', expert_d) 
