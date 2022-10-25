import numpy as np
import pandas as pd
import math
from itertools import combinations
df = pd.DataFrame(columns = ['p','ppg','s'])

cap = 60000
p = ['md','mr','cr','tg','jj','tg','dm','ra','hh','yk','js','cs','rg','bh','is']
ppg = [15.36,19.33,17.27,17.61,14.43,14.95,12.88,13.32, 8.57,11.5,9.71,6.91,7.94,5.37,1.69]
s = np.array([15, 14.5, 14,13.5,13,12,11,10, 9.5,9,8.5,8,7.5,7,7.5])*1000
s = s.tolist()

df['p'] = p
df['ppg'] = ppg
df['s'] = s

options = []
curr_row = np.array([0,0,0])


for row in df.iterrows():
    row = np.array(row[1])
    options.append(row)
    # curr_row+=row
    #
    # if curr_row[-1] > cap:
    #     options.append(curr_row)
    #     curr_row = np.array

maxs = []
for pe in [4,5,6,7]:
    perm = combinations(options, pe)
    newopt = []
    for i in perm:

        sal = 0
        for c in i:
            sal += c[-1]

        if sal <= cap:
            newopt.append(i)

    d = []
    pp = []

    for i in newopt:
        ppg = 0
        for c in i:
            ppg += c[1]
        d.append(i)
        pp.append(ppg)


    max = np.max(pp)
    maxind = pp.index(max)

    maxs.append([d[maxind], max])

asd = []
for kek in maxs:
    asd.append(kek[-1])

maxind = asd.index(np.max(asd))
opti = maxs[maxind]

print(opti)



