import numpy as np
import scipy.io
import os
import openpyxl
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.drift_detection.page_hinkley import PageHinkley
from helpers import mask_types_old
from random_perm_n import random_perm_n_old
from DCFHT import OCFHT_drift_new

dataset = ['wpbc']

for i in range(len(dataset)):
    file = '/DCFHT/data/' + dataset[i] + '.mat'
    mat_data = scipy.io.loadmat(file)
    data = mat_data['data']
    X = data[:, 1:data.shape[1]]
    X = X.astype(float)
    MASK_NUM = 1
    X_masked = mask_types_old(X, MASK_NUM, seed=1)  # arbitrary setting Nan；seed=1
    # X = X.values
    Y_label = data[:, 0]

    n = X_masked.shape[0]
    feat = X_masked.shape[1]
    Y_label = Y_label.flatten()

    is_drift = True

    acc = np.zeros((1, 10))
    runtime = np.zeros((1, 10))

    permutations = random_perm_n_old(n)
    for j in range(10):
        # print(j)
        perm = permutations[j]
        Y = Y_label[perm]
        X = X_masked[perm]
        Y = Y.astype(float)

        model = OCFHT_drift_new(HoeffdingTreeClassifier(), 0, PageHinkley(delta=0.01), PageHinkley(delta=0.005), False, is_drift)
        classifier1, err_count1, correct_cnt1, runtime1, acc_all = model.OCFHT_drift_OVFM(X, Y)
        acc[0, j] = correct_cnt1 / n
        runtime[0, j] = runtime1