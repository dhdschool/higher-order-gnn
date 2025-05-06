
from numba import config
config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY = True

from typing import Iterable
import numba.cuda as cuda
import numpy as np
import math
import numba as nb

print(cuda.is_available())


fact_list = [math.factorial(i) for i in range(21)]
FACT_TABLE = np.array(fact_list, dtype=np.int64)

@cuda.jit('int64(int32)', device=True)
def factorial(n: int):
    if n > 20:
        raise ValueError("Integer overflow for dtype int64 for factorial > 20")
    return FACT_TABLE[n]
            
@cuda.jit('int64(int32, int32)', device=True)
def comb(n: int, k: int):
    return int(factorial(n) / (factorial(k) * factorial(n-k)))

@cuda.jit('int32[:](int32[:], int32, int32, int32, int32)', device=True)
def combinations(x: np.ndarray, k: int, threads:int=0, depth: int=0, n:int=0):
    if n == 0:    
        n = len(x)
    
    if threads == 0:
        threads = comb(n, k)
    
    iterables = np.full((threads, k), fill_value=-1, dtype=np.int32)
    subthread_count = 0
    for idx in range(n - k + 1):
        subthreads = comb(n - idx - 1, k - 1)
        iterables[subthread_count:subthreads + subthread_count, 0] = np.repeat(x[idx + depth], subthreads)
        
        if iterables.shape[1] != 1:
            iterables[subthread_count:subthreads + subthread_count, 1:] = combinations(x, k-1, subthreads, depth+1+idx, n-idx-1)        
        subthread_count = subthreads + subthread_count
    return iterables

@cuda.jit('void(int32[:], int32)')
def kernel(x, k):
    return combinations(x, k)

#if __name__ == '__main__':
    #x = np.arange(6) + 1
    #print(kernel(x, 3))