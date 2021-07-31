from scipy import stats
import statistics as sts
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


expert_ppc_total = [109, 123, 13]
expert_ppc_resps = [14, 9, 4]
expert_m2_total = [85, 117]

naive_m2_total = [153, 20, 20, 98, 228, 154]
naive_ppc_total = [95, 58, 52, 97, 119, 139]

print(sts.median(naive_m2_total))
print(sts.median(naive_ppc_total))
print(sts.median(expert_m2_total))
print(sts.median(expert_ppc_total))
