# This file is meant as an alternative to the combinitorial complex class present in TopoX
# for the purpose of topological neural networks. This should ideally provide speedup
# for generating neighborhood matrices, especially when a CUDA hardware accelerator is
# avaliable

from typing import Iterable
import numba as nb
import numpy as np
import math

fact_list = [math.factorial(i) for i in range(21)]
FACT_TABLE = np.array(fact_list, dtype=np.int64)

@nb.jit
def factorial(n: int):
    if n > 20:
        raise ValueError("Integer overflow for dtype int64 for factorial > 20")
    return FACT_TABLE[n]
            
@nb.jit
def comb(n: int, k: int):
    return int(factorial(n) / (factorial(k) * factorial(n-k)))

# Order preserving combinations
@nb.jit
def combinations(x: np.ndarray, k: int, threads=None, depth: int=0, n=None):
    if n == None:    
        n = len(x)
    
    if threads == None:
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
        
            
    
def group(x0, x1) -> dict:
    # Map from x0 to x1
    pass
    # Generate subgroups of x1
    
if __name__ == '__main__':
    arr = np.arange(6) + 1
    print(combinations(arr, 4))