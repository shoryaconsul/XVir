#!/usr/bin/env python3.8

import os
import numpy as np
import bz2
import _pickle as cPickle

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

class Ngram(object):
    """
    Tokenize x into n-grams

    Takes a NDArray data instance and converts it into a sequence of N-grams.
    Output is 2D array where each column in an N-gram.
    """

    def __init__(self, n=3):
        assert isinstance(n, int)
        self.n = n  # Length of N-gram
    
    def __call__(self, sample):
        ngram_list = [sample[i:i+self.n] for i in range(len(sample)-self.n+1)]
        ngram_tensor = torch.stack(ngram_list).long()  # Each row is an N-gram
        base_tensor = torch.pow(4, torch.arange(3, dtype=torch.long))
        ngram_val_tensor = torch.matmul(ngram_tensor-1, base_tensor)  # Converting tensor to int

        # return F.one_hot(ngram_val_tensor, num_classes=4**n)
        return ngram_val_tensor

class kmerDataset(Dataset):
    """
    Tokenizing reads as a sequence of k-mers.
    We initially set k = 3.
    """

    def __init__(self, args, transform=Ngram):
        filename = os.path.join(args.data_path, args.data_file)
        self.transform = transform(args.ngram)

        with bz2.open(filename, 'rb') as f:
            self.file = cPickle.load(f)
        self.reads = self.file['reads']
        self.labels = self.file['labels']
        self.data = torch.from_numpy(self.reads)
        self.y = torch.from_numpy(self.labels).unsqueeze(-1).float()

    def __getitem__(self, index):
        """
            Return read and label for read
        """
        
        x = self.data[index, ...]
        label = self.y[index, ...]

        if self.transform:
            x = self.transform(x)

        return x, label

    def __len__(self):
        return self.y.shape[0]
