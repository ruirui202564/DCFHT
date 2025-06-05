import numpy as np
import pandas as pd
import math
import os

def chack_Nan(X_masked,n):
    for i in range(n):
        X_masked[i] = X_masked[i].strip()
        X_masked[i] = X_masked[i].strip("[]")
        X_masked[i] = X_masked[i].split(",")
        X_masked[i] = list(map(float, X_masked[i]))
        narry = np.array(X_masked[i])
        where_are_nan = np.isnan(narry)
        narry[where_are_nan] = 0
        X_masked[i] = narry.tolist()
    return X_masked

def get_cont_indices(X):
    max_ord=14
    indices = np.zeros(X.shape[1]).astype(bool)
    for i, col in enumerate(X.T):
        col_nonan = col[~np.isnan(col)]
        col_unique = np.unique(col_nonan)
        if len(col_unique) > max_ord:
            indices[i] = True
    return indices

def cont_to_binary(x):
    # make the cutoff a random sample and ensure at least 10% are in each class
    while True:
        cutoff = np.random.choice(x)
        if len(x[x < cutoff]) > 0.1*len(x) and len(x[x < cutoff]) < 0.9*len(x):
            break
    return (x > cutoff).astype(int)

def cont_to_ord(x, k):
    # make the cutoffs based on the quantiles
    std_dev = np.std(x)
    cuttoffs = np.linspace(np.min(x), np.max(x), k+1)[1:]
    ords = np.zeros(len(x))
    for cuttoff in cuttoffs:
        ords += (x > cuttoff).astype(int)
    return ords.astype(int)

def mask_types_old(X, mask_num, seed):
    X_masked = np.copy(X)
    mask_indices = []
    num_rows = X_masked.shape[0]
    num_cols = X_masked.shape[1]
    for i in range(num_rows):
        np.random.seed(seed*num_rows-i)
        for j in range(num_cols//2):
            rand_idx=np.random.choice(2,mask_num,False)
            for idx in rand_idx:
                X_masked[i,idx+2*j]=np.nan
                mask_indices.append((i, idx+2*j))
    return X_masked

def mask_types(X, mask_num, seed):
    X_masked = np.copy(X)
    mask_indices = []
    num_rows = X_masked.shape[0]
    num_cols = X_masked.shape[1]
    for i in range(num_rows):
        np.random.seed(seed*num_rows-i)
        num_mask = int(mask_num * num_cols)
        rand_idx = np.random.choice(num_cols, num_mask, replace=False)
        for idx in rand_idx:
            X_masked[i, idx] = np.nan
            mask_indices.append((i, idx))
    return X_masked