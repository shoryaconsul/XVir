import sys
import os
from tqdm import tqdm

import torch
from torch.nn import functional as F
from model import XVir
from utils.dataset import kmerDataset

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

torch.manual_seed(4)

class Args:
    def __init__(self):
        self.read_len = 150
        self.ngram = 6
        self.model_dim = 128
        self.num_layers = 1
        self.dropout = 0.1
        self.batch_size = 100
        # self.model_path="logs/experiment/XVir_models/XVir-2023.10.27-01-08-29/XVir-2023.10.27-01-08-29_2023.10.27-04-07-36.pt"
        # self.model_path = 'logs/experiment/XVir_models/XVir-2023.07.13-01-19-21/'\
        #     'XVir-2023.07.13-01-19-21_2023.07.13-03-37-18.pt'
        self.model_path = 'logs/experiment/XVir_models/XVir-2024.03.24-19-58-12/XVir-2024.03.24-19-58-12_2024.03.25-00-57-34.pt'
        self.data_path = 'data/split'
        self.data_file = 'test_data.pkl'


activation = {}
def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


args = Args()
# load model
model = XVir(args.read_len, args.ngram, args.model_dim, args.num_layers, args.dropout)
model.load_state_dict(torch.load(args.model_path))

# load data
test_dataset = kmerDataset(args)
test_loader = torch.utils.data.DataLoader(test_dataset,
    batch_size=args.batch_size, shuffle=True)

model.eval()
lin_weights = model.prob[1].weight.data
lin_mask = lin_weights.unflatten(1,
        (args.read_len - args.ngram + 1, args.model_dim)
        ).squeeze(0)
lin_bias = model.prob[1].bias.data
print(lin_weights, lin_bias)

h = model.xformEncoder.register_forward_hook(getActivation('xformEncoder'))
attn_list = []
true_label_list = []
pred_list = []
with torch.no_grad():
    for x_val, y_val in tqdm(test_loader):
        # print(x_val.size(), y_val.squeeze(1).size())
        true_label_list.append(y_val.squeeze(1))
        outputs = model(x_val)
        pred = torch.sigmoid(outputs.detach()).to('cpu')
        pred_list.append(pred[0])
        attn_list.append(activation['xformEncoder'])
h.remove()

# Collate results
true_labels = torch.cat(true_label_list)
pred_probs = torch.cat(pred_list)
read_attns = torch.vstack(attn_list)
# read_attns_avg = torch.mean(read_attns, dim=2)  #*lin_mask
read_attns_avg = read_attns
attn_min = torch.min(read_attns_avg)
attn_max = torch.max(read_attns_avg)
print('Range of attention values: [%.3f, %.3f]' %(attn_min, attn_max))

# Randomly select subset of reads to show
num_show = 5  # number of reads to show
viral_mask = torch.nonzero(true_labels == 1, as_tuple=True)
nonviral_mask = torch.nonzero(true_labels == 0, as_tuple=True)
viral_read_attns = read_attns_avg[viral_mask]
nonviral_read_attns = read_attns_avg[nonviral_mask]
viral_sel_idx = torch.randperm(torch.sum(true_labels == 1))[:num_show]
nonviral_sel_idx = torch.randperm(torch.sum(true_labels == 0))[:num_show]

viral_read_attns_sel = viral_read_attns[viral_sel_idx].squeeze(1)
nonviral_read_attns_sel = nonviral_read_attns[nonviral_sel_idx].squeeze(1)
all_read_attns = torch.vstack((viral_read_attns_sel,nonviral_read_attns_sel)).numpy()

fig, ax = plt.subplots(3,num_show,figsize=(25,10))
read_attns_arr = torch.flatten(read_attns).numpy()
read_attn_mean = np.mean(read_attns_arr)  # Stats of read attentions
read_attn_std = np.std(read_attns_arr)
read_attn_min = read_attn_mean - 2*read_attn_std
read_attn_max = read_attn_mean + 2*read_attn_std
for i in range(num_show):
    ax[0,i].imshow(viral_read_attns_sel[i]*lin_mask,
                vmin=read_attn_min, vmax=read_attn_max,
                cmap='plasma', aspect='auto')
    ax[1,i].imshow(nonviral_read_attns_sel[i]*lin_mask,
                vmin=read_attn_min, vmax=read_attn_max,
                cmap='plasma', aspect='auto')
    ax[2,i].imshow(viral_read_attns_sel[i]*lin_mask - 
                nonviral_read_attns_sel[i]*lin_mask,
                vmin=read_attn_min, vmax=read_attn_max,
                cmap='plasma', aspect='auto')
    # if i == 0:
    #     print(torch.sigmoid(viral_read_attns_sel[i]*lin_mask) -
    #             torch.sigmoid(nonviral_read_attns_sel[i]*lin_mask))
plt.savefig('attns.jpg')

# Histogram of read attentions
plt.figure(figsize=(15,10))
print('Mean attention: %.3f, std: %.3f' %(
    np.mean(read_attns_arr), np.std(read_attns_arr)))
plt.hist(read_attns_arr, 100, range=(-1,1))
plt.savefig('attns_hist.jpg')


# Showing linear mask
fig, ax = plt.subplots(1,2,figsize=(25,10))
im1 = ax[0].imshow(lin_mask, cmap='plasma', aspect='auto')
im2 = ax[1].imshow(torch.abs(lin_mask) > torch.std(lin_mask),
    cmap='plasma', aspect='auto')
plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])
plt.savefig('lin_mask.jpg')
# print("--------------------")
# print(np.nonzero(torch.abs(lin_mask) > torch.std(lin_mask)))

## Plotting attention values
# print(viral_read_attns.size(), nonviral_read_attns.size(), lin_mask.size())
viral_read_attns = torch.mul(viral_read_attns, lin_mask)
nonviral_read_attns = torch.mul(nonviral_read_attns, lin_mask)

fig, ax = plt.subplots(1,2,figsize=(25,10))
vmin, vmax = (read_attns*lin_mask + lin_bias).min(), (read_attns*lin_mask + lin_bias).max()
im1 = ax[0].imshow(torch.mean(viral_read_attns*lin_mask, axis=0), cmap='bwr', aspect='auto',
                vmin=vmin, vmax=vmax)
im2 = ax[1].imshow(torch.mean(nonviral_read_attns*lin_mask, axis=0), cmap='bwr', aspect='auto',
                vmin=vmin, vmax=vmax)
plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])
plt.savefig('attns_viral_nonviral.jpg')


# Plotting significant entries of attention values
def linear(attns):
    return attns*lin_mask + lin_bias

read_sig_min = torch.min(linear(read_attns))
read_sig_max = torch.max(linear(read_attns))
read_sig_sd = torch.std(linear(read_attns))
read_sig_avg = torch.mean(linear(read_attns))
print('Mean, std of linear attns: %.3f, %.3f' %(read_sig_avg, read_sig_sd))
viral_idx, viral_pos, viral_latent = torch.nonzero(
    np.abs(linear(viral_read_attns) - read_sig_avg) > read_sig_sd, as_tuple=True)
nonviral_idx, nonviral_pos, nonviral_latent = torch.nonzero(
    np.abs(linear(nonviral_read_attns) - read_sig_avg) > read_sig_sd, as_tuple=True)

print('VIRAL: ---------------------')
print(viral_pos, viral_latent)
print('Significant fraction: %.3f' %(len(viral_pos)/torch.numel(viral_read_attns)))
print('NONVIRAL: ---------------------')
print(nonviral_pos, nonviral_latent)
print('Significant fraction: %.3f' %(len(nonviral_pos)/torch.numel(nonviral_read_attns)))

# fig, ax = plt.subplots(1, 2,  figsize=(25,10))
# ax[0].scatter(viral_pos, viral_latent, s=1, c='r')
# ax[1].scatter(nonviral_pos, nonviral_latent, s=1, c='b')
# plt.savefig('read_sig.jpg')

viral_sigs = torch.zeros_like(viral_read_attns[0])
for attn in viral_read_attns:
    viral_sigs += (torch.abs(linear(attn) - read_sig_avg) > read_sig_sd)
viral_sigs = viral_sigs / viral_read_attns.size(0)

nonviral_sigs = torch.zeros_like(nonviral_read_attns[0])
for attn in nonviral_read_attns:
    nonviral_sigs += (torch.abs(linear(attn) - read_sig_avg) > read_sig_sd)
nonviral_sigs = nonviral_sigs / nonviral_read_attns.size(0)

fig, ax = plt.subplots(1, 3,  figsize=(30,10))
ax[0].imshow(viral_sigs, aspect='equal', cmap='bwr', vmin=-1, vmax=1)
ax[1].imshow(nonviral_sigs, aspect='equal', cmap='bwr', vmin=-1, vmax=1)
img3 = ax[2].imshow(viral_sigs - nonviral_sigs, aspect='equal', cmap='bwr',
                vmin=-0.25, vmax=0.25)
                # vmin=-2*read_sig_sd, vmax=2*read_sig_sd)
# plt.colorbar(img3, ax=ax[2])
plt.savefig('read_sig_img.jpg')


# Plotting histogram of read attentions - mean + SD
# viral_read_attns_hist = F.softmax(
#     torch.sum(viral_read_attns, dim=2), dim=1)
viral_read_attns_hist = torch.sum(viral_read_attns, dim=2)
viral_read_attns_sd, viral_read_attns_mean = torch.std_mean(
    viral_read_attns_hist, dim=0)

# nonviral_read_attns_hist = F.softmax(
#     torch.sum(nonviral_read_attns, dim=2), dim=1)
nonviral_read_attns_hist = torch.sum(nonviral_read_attns, dim=2)
nonviral_read_attns_sd, nonviral_read_attns_mean = torch.std_mean(
    nonviral_read_attns_hist, dim=0)

# print('Viral read stats')
# print(list(zip(viral_read_attns_mean, viral_read_attns_sd)))
# print('------------------------')
# print('Non-viral read stats')
# print(list(zip(nonviral_read_attns_mean, nonviral_read_attns_sd)))
# print('------------------------')

kmer_pos = np.arange(args.read_len - args.ngram + 1)
baseline = 1/len(kmer_pos)  # Uniform distribution

fig, ax = plt.subplots(2, 1,  figsize=(12,12))
ax[0].bar(kmer_pos, np.abs(viral_read_attns_mean),
        alpha=0.8, color='b', label='Viral')
# ax[0].errorbar(kmer_pos, viral_read_attns_mean,
#             yerr=viral_read_attns_sd, color="r", alpha=0.5)
ax[0].set_title('Viral', fontsize=24)
ax[1].bar(kmer_pos, np.abs(nonviral_read_attns_mean),
        alpha=0.8, color='g', label='Non-viral')
# ax[1].errorbar(kmer_pos, nonviral_read_attns_mean,
#             yerr=nonviral_read_attns_sd, color="r", alpha=0.5)
ax[1].set_title('Non-viral', fontsize=24)

ax[0].set_ylim([baseline, 0.4])
ax[0].tick_params(labelsize=20)
ax[0].set_ylabel('Absolute value', fontsize=24)
ax[1].set_ylim([baseline, 0.4])
ax[1].tick_params(labelsize=20)
ax[1].set_ylabel('Absolute value', fontsize=24)
ax[1].set_xlabel('Position', fontsize=24)
plt.savefig('read_attns_hist.jpg')

kmer_pos = np.arange(args.read_len - args.ngram + 1)
baseline = 1/len(kmer_pos)  # Uniform distribution

# Same plot with shared x-axis and real value
fig, ax = plt.subplots(1, 1,  figsize=(12,12))
ax.bar(kmer_pos, viral_read_attns_mean,
        alpha=0.8, color='b', label='Viral')
# ax[0].errorbar(kmer_pos, viral_read_attns_mean,
#             yerr=viral_read_attns_sd, color="r", alpha=0.5)
ax.bar(kmer_pos, nonviral_read_attns_mean,
        alpha=0.8, color='g', label='Non-viral')
# ax[1].errorbar(kmer_pos, nonviral_read_attns_mean,
#             yerr=nonviral_read_attns_sd, color="r", alpha=0.5)

ax.set_ylim([-0.4, 0.4])
ax.tick_params(labelsize=20)
ax.set_ylabel('Contribution to logit', fontsize=24)
ax.set_xlabel('Position', fontsize=24)
plt.legend(fontsize=20)
plt.savefig('read_attns_hist2.jpg')


# Plotting histograms taking most significant k-mers
viral_best_kmers = torch.argmax(viral_read_attns_hist, dim=1)
nonviral_best_kmers = torch.argmin(nonviral_read_attns_hist, dim=1)
print('BEST POSITIONS: ---------------------')
print(viral_best_kmers)
print(nonviral_best_kmers)

kmer_pos = torch.arange(args.read_len - args.ngram + 1)
baseline = 1/len(kmer_pos)  # Uniform distribution
viral_idx, viral_cnt = torch.unique(viral_best_kmers, return_counts=True)
viral_hist_cnt = torch.zeros_like(kmer_pos)
viral_hist_cnt[viral_idx] = viral_cnt

nonviral_idx, nonviral_cnt = torch.unique(nonviral_best_kmers, return_counts=True)
nonviral_hist_cnt = torch.zeros_like(kmer_pos)
nonviral_hist_cnt[nonviral_idx] = nonviral_cnt

# print(viral_cnt)
# print(nonviral_cnt)

fig, ax = plt.subplots(2, 1,  figsize=(12,12))
ax[0].bar(kmer_pos, viral_hist_cnt/torch.sum(viral_hist_cnt),
        alpha=0.8, color='b', label='Viral')
ax[0].set_title('Viral', fontsize=24)
ax[1].bar(kmer_pos, nonviral_hist_cnt/torch.sum(nonviral_hist_cnt),
        alpha=0.8, color='g', label='Non-viral')
ax[1].set_title('Non-viral', fontsize=24)

# ax[0].set_ylim(bottom=baseline, top=0.045)
# ax[1].set_ylim(bottom=baseline, top=0.045)
ax[0].tick_params(labelsize=20)
ax[1].tick_params(labelsize=20)
ax[1].set_xlabel('Position', fontsize=24)
plt.savefig('read_kmer_hist.jpg')


# # Median + quantiles
# quantiles = torch.Tensor([0.25, 0.5, 0.75])
# viral_read_attns_quantiles = torch.quantile(viral_read_attns_hist, quantiles, dim=0)
# nonviral_read_attns_quantiles = torch.quantile(nonviral_read_attns_hist, quantiles, dim=0)
# viral_read_attns_err = torch.vstack(
#     (viral_read_attns_quantiles[1] - viral_read_attns_quantiles[0],
#     viral_read_attns_quantiles[2] - viral_read_attns_quantiles[1])
#     )
# nonviral_read_attns_err = torch.vstack(
#     (nonviral_read_attns_quantiles[1] - nonviral_read_attns_quantiles[0],
#     nonviral_read_attns_quantiles[2] - nonviral_read_attns_quantiles[1])
#     )

# print('Viral read quantiles')
# print(viral_read_attns_quantiles)
# print('------------------------')
# print('Non-viral read quantiles')
# print(nonviral_read_attns_quantiles)
# print('------------------------')

# kmer_pos = np.arange(args.read_len - args.ngram + 1)
# baseline = 1/len(kmer_pos)  # Uniform distribution

# fig, ax = plt.subplots(2, 1,  figsize=(12,18))
# ax[0].bar(kmer_pos, viral_read_attns_quantiles[1],
#         alpha=0.8, color='b', label='Viral')
# ax[0].errorbar(kmer_pos, viral_read_attns_quantiles[1],
#             yerr=viral_read_attns_err, color="r", alpha=0.5)
# ax[0].set_title('Viral', fontsize=24)
# ax[1].bar(kmer_pos, nonviral_read_attns_quantiles[1],
#         alpha=0.8, color='g', label='Non-viral')
# ax[1].errorbar(kmer_pos, nonviral_read_attns_mean,
#             yerr=nonviral_read_attns_err, color="r", alpha=0.5)
# ax[1].set_title('Non-viral', fontsize=24)

# ax[0].set_ylim(bottom=0)
ax[0].tick_params(labelsize=20)
# ax[1].set_ylim(bottom=0)
ax[1].tick_params(labelsize=20)

## Single plot
# plt.figure(figsize=(12,8))
# ax = plt.gca()
# ax.bar(kmer_pos, viral_read_attns_mean, width=1.5,
#     alpha=0.6, color='b', label='Viral')
# ax.bar(kmer_pos, nonviral_read_attns_mean, width=1.5,
#     alpha=0.6, color='g', label='Non-viral')

# ax.set_ylim(bottom=baseline)
# ax.legend(prop={'size': 20})
# ax.tick_params(labelsize=20)
# plt.savefig('read_attns_hist.jpg')




# Heatmap of correlations of read attentions
viral_read_attns_vec = viral_read_attns.view(
    viral_read_attns.size(0), -1)
nonviral_read_attns_vec = nonviral_read_attns.view(
    nonviral_read_attns.size(0), -1)
read_attns_vec = torch.cat((viral_read_attns_vec,
                    nonviral_read_attns_vec))
read_attns_corr = torch.flip(torch.corrcoef(read_attns_vec), dims=[0])

plt.figure(figsize=(10,10))
plt.imshow(read_attns_corr, cmap='seismic', vmin=-1, vmax=1,
            extent=[0, len(read_attns_corr), 0, len(read_attns_corr)])
plt.gca().tick_params(labelsize=20)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.savefig('read_attns_corr.jpg')