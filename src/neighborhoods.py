# This file is meant as an alternative to the combinitorial complex class present in TopoX
# for the purpose of topological neural networks. This should ideally provide speedup
# for generating neighborhood matrices, especially when a CUDA hardware accelerator is
# avaliable

from typing import Iterable
import numba as nb
import numpy as np
import math
from time import time
import torch

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

def construct_map(x0: np.ndarray, x1: np.ndarray):
    subgroup_map = {}
    k = x0.shape[1]
    
    for x in x1:
        for subgroup in combinations(x, k):
            key = tuple(subgroup)
            subgroup_map[key] = subgroup_map.get(key, []) + [x]
    return subgroup_map

def construct_idx_map(x0: np.ndarray, x1: np.ndarray):
    index_map = {}
    k = x0.shape[1]
    
    x1_indices = np.arange(len(x1))
    for x, x1_idx in zip(x1, x1_indices):
        for subgroup in combinations(x, k):
            key = tuple(subgroup)
            index_map[key] = index_map.get(key, []) + [x1_idx]
    return index_map

def incidence_matrix(x0, x1):
    n = len(x0)
    m = len(x1)
    
    index_map = construct_idx_map(x0, x1)
    
    idx_0 = []
    idx_1 = []
    values = []
    
    for group, x0_idx in zip(x0, np.arange(n)):
        key = tuple(group)
        for x1_idx in index_map.get(key, []):
            idx_0.append(x0_idx)
            idx_1.append(x1_idx)
            values.append(1)
    indices = [idx_0, idx_1]
    
    matrix = torch.sparse_coo_tensor(indices, values, (n,m))    
    return matrix

if __name__ == '__main__':
    from dataset import TnnDataset, Reader
    reader = Reader('data/ciao')
    data = TnnDataset(reader)
    x0 = data.x1
    x1 = data.x2[0]
    
    start_time = time()
    mat = incidence_matrix(x0, x1)
    print(mat)
    end_time = time()
    print(end_time - start_time)