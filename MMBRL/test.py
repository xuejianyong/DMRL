import configparser
import os
import numpy as np

config = configparser.ConfigParser()
config.read('../../app.ini')
TRIAL_SAMEJIMA = config.getint('env','trial_samejima')
N_MODULE = config.getint('env','n_modules')
# print(TRIAL_SAMEJIMA)
# print(N_MODULE)
# print(os.getcwd())

a = [[1,2,3],
     [4,5,6],
     [1,2,3],
     [2,3,7]]

mean_value = np.mean(a, axis=0)
print(mean_value)

print(12/5)

a = {}
a['a'] = 0.1
a['b'] = 0.2
a['c'] = 0.3
print(max(a.values()))