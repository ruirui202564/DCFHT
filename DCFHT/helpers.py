import numpy as np
import pandas as pd
import math
import os

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
