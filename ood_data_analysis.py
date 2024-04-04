# This script analyzes the performance of XVir on out-of-distribution (OOD) data.

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
        self.model_path = 'logs/experiment/XVir_models/XVir-2023.07.13-01-19-21/'\
            'XVir-2023.07.13-01-19-21_2023.07.13-03-37-18.pt'
        self.data_path = 'data/split'
        self.data_file = 'test_data.pkl'

    model = XVir(
        args.read_len, args.ngram, args.model_dim, args.num_layers, args.dropout
    )