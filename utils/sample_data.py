## Sample reads from training data

import numpy as np
import os
import bz2
import _pickle as cPickle
import argparse


def parse_fasta(filename):
    base2int = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}

    reads = []
    labels = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:  # EOF
                break
            elif line[0] == '>':  # Parse sequence data
                read_header = line[1:].rstrip()
                read_label = 1 - int('HUM' in read_header)
                labels.append(read_label)
                read_seq = f.readline().rstrip().upper()
                read_seq = np.array([base2int[base] for base in read_seq])
                reads.append(read_seq)
            else:
                pass # For completeness
    return np.array(reads), np.array(labels)


parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('--num-reads', type=int, default=1000,
                    help='Number of reads to sample')
parser.add_argument('--out-path', type=str,
                default='data/proc_data_sample.pkl',
                help='Path to save sampled data')
parser.add_argument('--seed',   type=int, default=4, help='Random seed')

args = parser.parse_args()
np.random.seed(args.seed)

# Load data

filename = 'data/split/train_data.fa'
ext = os.path.splitext(filename)[1]
if ext == '.pkl':
    with bz2.open(filename, 'rb') as f:
        file = cPickle.load(f)
        reads = file['reads']
        labels = file['labels']
elif ext in ['.fa', '.fasta']:
    reads, labels = parse_fasta(filename)
else:
    raise ValueError('Data file must be a Pickle or FASTA file.')

# Sample reads
sample_idx = np.random.choice(len(reads), args.num_reads,
                            replace=False)
# sample_idx = np.random.choice(len(reads), len(reads),
#                             replace=False)
sample_reads = reads[sample_idx]
sample_labels = labels[sample_idx]

sample_data_dict = {'reads': sample_reads,
                    'labels': sample_labels}
with bz2.BZ2File(args.out_path,'wb') as wfile:
    cPickle.dump(sample_data_dict, wfile, protocol=4)

