import numpy as np
import matplotlib.pyplot as plt


def calc_entropy(dist):
    my_sum = 0
    for p in dist:
        if p > 0:
            my_sum += p * np.log(p) / np.log(len(dist))
    return - my_sum


def calc_relative_entropy(dist1, dist2):
    my_sum = 0
    for p, q in zip(dist1, dist2):
        if p > 0 and q > 0:
            my_sum += p * np.log(p/q) / np.log(len(dist1))
    return my_sum


def calc_extropy(dist):
    my_sum = 0
    for p in dist:
        p = 1 - p
        if p > 0:
            my_sum += p * np.log(p) / np.log(len(dist))
    return - my_sum


random_dists = np.random.rand(100, 5)

row_sums = random_dists.sum(axis=1, keepdims=True)
normed_dists = random_dists / row_sums

entropies = []
extropies = []

for idx in range(normed_dists.shape[0]):
    entropies.append(calc_entropy(normed_dists[idx]))
    extropies.append(calc_extropy(normed_dists[idx]))


plt.figure(1)
plt.scatter(entropies, extropies)

plt.show()

