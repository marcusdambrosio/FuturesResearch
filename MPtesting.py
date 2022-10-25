import multiprocessing as mp
import itertools as iter
from EMAcrossover import simulate
import time


shorts = [9, 14 , 20 , 30, 50]
longs = [20, 30 ,50 ,100]

pairs = []

for s in shorts:
    for l in longs:
        if l > s: pairs.append(['NQU20', '5min', s,l, 10])


numthreads = 8

pool, out = mp.Pool(numthreads), []

outputs = pool.imap_unordered(simulate, pairs)

pool.close();pool.join()


start = time.perf_counter()




finish = time.perf_counter()