# Visualizating data

import numpy as np
import os
import bz2
import _pickle as cPickle
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.manifold import TSNE, MDS
from matplotlib import pyplot as plt
import matplotlib as mpl

font = {'size' : 20,
        'family': 'sans-serif'}
mpl.rc('font', **font)
mpl.rc('axes', labelsize=24)

np.random.seed(4)

## Checking viral and non-viral Hamming distances
with bz2.open('data/proc_data.pkl', 'rb') as f:
    f = cPickle.load(f)
    reads = f['reads']
    labels = f['labels']

print('Loaded reads')

num_samples = 1000  # Sampling to reduce computation complexity
nv_idx = np.random.choice(np.sum(labels == 0), num_samples, replace=False)
v_idx = np.random.choice(np.sum(labels == 1), num_samples, replace=False)

nv_reads = reads[labels == 0][nv_idx]
v_reads = reads[labels == 1][v_idx]

v_nv_reads = np.vstack([v_reads, nv_reads])
read_dist = squareform(pdist(v_nv_reads, metric='hamming'))

# Multi-dimensional scaling
mds = MDS(n_components=2, dissimilarity='precomputed', normalized_stress=False)
mds_embed = mds.fit_transform(read_dist)

plt.figure(figsize=(10, 10))
plt.scatter(mds_embed[:num_samples, 0], mds_embed[:num_samples, 1],
        c='tab:blue', label='Non-viral')
plt.scatter(mds_embed[num_samples:, 0], mds_embed[num_samples:, 1],
        c='tab:orange', label='Viral')
plt.legend()
plt.savefig('data/mds.jpg')

# t-SNE
tsne = TSNE(n_components=2, metric='precomputed', init='random')
tsne_embed = tsne.fit_transform(read_dist)
plt.figure(figsize=(10, 10))
plt.scatter(tsne_embed[:num_samples, 0], tsne_embed[:num_samples, 1],
        c='tab:blue', label='Non-viral')
plt.scatter(tsne_embed[num_samples:, 0], tsne_embed[num_samples:, 1],
        c='tab:orange', label='Viral')
plt.legend()
plt.savefig('data/tsne.jpg')

# nv_dist = pdist(nv_reads, metric='hamming')
# v_dist = pdist(v_reads, metric='hamming')
# v_nv_dist = cdist(v_reads, nv_reads, metric='hamming')

# print('Mean intra non-viral distance: ', np.mean(nv_dist))
# print('Mean intra viral distance: ', np.mean(v_dist))
# print('Mean inter viral-non-viral distance: ', np.mean(v_nv_dist))