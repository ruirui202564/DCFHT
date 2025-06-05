import numpy as np

def random_perm_n(n):
    np.random.seed(1)
    permutations = []
    for i in range(10):
        perm = np.arange(n)
        if i == 0:
            permutations.append(perm)
        else:
            np.random.shuffle(perm)
            permutations.append(perm)

    return permutations

def random_perm_n_old(n):
    np.random.seed(1)
    permutations = []
    for _ in range(10):
        perm = np.arange(n)
        np.random.shuffle(perm)
        permutations.append(perm)

    return permutations