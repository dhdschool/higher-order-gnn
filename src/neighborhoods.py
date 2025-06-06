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
        #raise ValueError("Integer overflow for dtype int64 for factorial > 20")
        return -1
    return FACT_TABLE[n]

@nb.jit
def partial_factorial(n, denominator):
    total = 1
    for i in range(denominator+1, n+1):
        total *= i
    return total    
          
@nb.jit
def comb(n: int, k: int):
    a = factorial(n)
    b = factorial(k)
    c = factorial(n-k)
    
    if a != -1 and b != -1 and c != -1:
        return int(factorial(n) / (factorial(k) * factorial(n-k)))
    else:
        return comb_2(n, k)
        
@nb.jit
def comb_2(n: int, k: int):
    if n-k > k:
        return int(partial_factorial(n, n-k) / factorial(k))
    else:
        return int(partial_factorial(n, k) / factorial(n-k))

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

# Order preserving non-recursive implementation of combinations, faster than combinations when n choose k is sufficiently small
@nb.jit
def combinations_2(x: np.ndarray, k: int):
    n = len(x)
    m = comb(n, k)
    
    arr = np.empty((m, k), dtype=np.int32)
    ptrs = np.arange(k)
    for idx in range(m):
         arr[idx, :] = x[ptrs]
         ptrs[-1] += 1
         passed_check = False
         while passed_check == False and idx != m-1:
            for ptr_idx in range(len(ptrs)-1, -1, -1):                                
                if ptrs[ptr_idx] > n - (k - ptr_idx):
                    ptrs[ptr_idx-1] = ptrs[ptr_idx-1] + 1
                    ptrs[ptr_idx:] = np.arange(ptrs[ptr_idx-1]+1, ptrs[ptr_idx-1]+(k-ptr_idx+1))
                    break
                if ptr_idx == 0:
                    passed_check=True
                
    return arr

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
        for subgroup in combinations_2(x, k):
            key = tuple(subgroup)
            index_map[key] = index_map.get(key, []) + [x1_idx]
    return index_map


@nb.jit
def construct_comb_arr(x1, k):
    entries = x1.shape[0]
    n = x1.shape[1]
    n_choose_k = comb(n, k)
    m = entries * n_choose_k
    
    combination_arr = np.empty((m, k))
    
    for idx, group in enumerate(x1):
        l_ptr = n_choose_k * idx
        r_ptr = n_choose_k * (idx + 1)
        combination_arr[l_ptr:r_ptr, :] = combinations(group, k) 
        
    return combination_arr

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

def adjacency_matrix(x0, x1):
    n = len(x0)
    
    idx_map = construct_idx_map(x0, x1)
    new_map = {}
    
    # Group all x0 indices by pairing with x1
    for i in range(len(x0)):
        key = tuple(x0[i])
        if key in idx_map:
            for x1_idx in idx_map[key]:
                new_map[x1_idx] = new_map.get(x1_idx, []) + [i]
    
    row_indices = []
    column_indices = []
    values = [] 
    
    # Find all 2 length combinations of adjacent x0 indices and fill in their value on the sparse tensor
    for keys in new_map.values():
        pairwise_indices = combinations_2(np.array(keys), 2)
        for row in pairwise_indices:
            row_indices.append(row[0])
            column_indices.append(row[1])
            values.append(1)
    
    indices = [row_indices, column_indices]
    matrix = torch.sparse_coo_tensor(indices, values, (n,n))    
    return matrix

def coadjacency_matrix(x0, x1):
    m = len(x1)
    idx_map = construct_idx_map(x0, x1)
    
    row_indices = []
    column_indices = []
    values = [] 
    
    # Group all x0 indices by pairing with x1
    for key in x0:
        key = tuple(key)
        if key in idx_map:
            x1_idx_list = idx_map[key]
            pairwise_indices = combinations_2(np.array(x1_idx_list), 2)
            for row in pairwise_indices:
                row_indices.append(row[0])
                column_indices.append(row[1])
                values.append(1)
    
    indices = [row_indices, column_indices]
    matrix = torch.sparse_coo_tensor(indices, values, (m,m))    
    return matrix
    
if __name__ == '__main__':
    d_start_time = time()
    from dataset import TnnDataset, Reader
    reader = Reader('data/ciao')
    data = TnnDataset(reader)
    x0 = data.x1
    x1 = data.x2[0]
    d_end_time = time()
    
    print(f"Dataset load time is {d_end_time - d_start_time}")
    
    start_time = time()
    
    print(coadjacency_matrix(x0, x1))
    end_time = time()
    print(end_time - start_time)