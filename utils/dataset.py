#!/usr/bin/env python3.8

import os
import numpy as np
import bz2
import _pickle as cPickle
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, Subset
from torch.nn import functional as F


def store_data_split(filebase, dataset, name):
    """
    Store test, validation and test data splits in FASTA files
    """

    base_arr = np.array(['N', 'A', 'C', 'G', 'T'])
    if not os.path.exists(filebase):
        os.makedirs(filebase)

    with open(os.path.join(filebase, name + '_data.fa'), 'w') as f:

        if not isinstance(dataset, Dataset):
            raise ValueError('dataset must be a Dataset or Subset object.')
        elif isinstance(dataset, Subset):
            reads = dataset.dataset.reads[np.array(dataset.indices)]
            labels = dataset.dataset.labels[np.array(dataset.indices)]
        else:
            reads = dataset.reads
            labels = dataset.labels

        for x, y in tqdm(zip(reads, labels), total=len(dataset)):
            if int(y) == 1:
                read_header = '>HPVREF|' + str(np.random.randint(100))
            else:
                read_header = '>HUMREF|' + str(np.random.randint(100))
            f.write(read_header + '\n')
            f.write(''.join(base_arr[x.astype(int)]) + '\n')


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
        base_tensor = torch.pow(4, torch.arange(self.n, dtype=torch.long))
        ngram_val_tensor = torch.matmul(ngram_tensor-1, base_tensor)  # Converting tensor to int

        # return F.one_hot(ngram_val_tensor, num_classes=4**n)
        return ngram_val_tensor


class kmerDataset(Dataset):
    """
    Tokenizing reads as a sequence of k-mers.
    We initially set k = 3.
    """

    def __init__(self, args, transform=Ngram, split=None):
        if not split:
            filename = os.path.join(args.data_path, args.data_file)
        elif split == 'train':
            filename = os.path.join(args.data_path, args.train_data_file)
        elif split == 'val':
            filename = os.path.join(args.data_path, args.val_data_file)
        elif split == 'test':
            filename = os.path.join(args.data_path, args.test_data_file)
        else:
            raise ValueError('Invalid split name.')

        if transform:
            self.transform = transform(args.ngram)
        else:
            self.transform = None

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

    def num_class(self):
        return len(np.unique(self.labels))

    def dimerize(self):
        """
        Augment the dataset by adding reverse complement of each read.
        The label is preserved as one would expect the reverse complement
        of a read to belong to the same class.
        """

        revcomp_reads = np.fliplr(5 - self.reads)
        self.reads = np.vstack((self.reads, revcomp_reads))
        self.data = torch.from_numpy(self.reads)
        self.labels = np.tile(self.labels, 2)
        self.y = torch.from_numpy(self.labels).unsqueeze(-1).float()


class inferenceDataset(Dataset):
    def __init__(self, args, reads, labels):
        self.transform = Ngram(args.ngram)
        self.data = torch.from_numpy(reads)
        self.labels = labels
        self.length = len(reads)

    def __getitem__(self, index):
        """
            Return read for index
        """
        x = self.data[index, ...]

        if self.transform:
            x = self.transform(x)

        return x, self.labels[index]
    
    def __len__(self):
        return self.length
