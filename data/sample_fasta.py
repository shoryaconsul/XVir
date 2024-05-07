## Augment training data by adding mutated reads to training FASTA file

import numpy as np
import os
import bz2
import _pickle as cPickle
import argparse


def parse_fasta(filename):
    reads = []
    read_headers = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:  # EOF
                break
            elif line[0] == '>':  # Parse sequence data
                read_header = line[1:].rstrip()
                read_seq = f.readline().rstrip().upper()
                read_headers.append(read_header)
                reads.append(read_seq)
            else:
                pass # For completeness
    return np.array(read_headers), np.array(reads)


parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('filename', type=str, help='Path to load data')
parser.add_argument('-o', '--out-path', type=str,
                help='Path to save sampled data')
parser.add_argument('-a','--aug-frac', type=float, default=0.1,
                    help='Ratio of augmented data to original training data')
parser.add_argument('--seed',   type=int, default=4, help='Random seed')

args = parser.parse_args()
np.random.seed(args.seed)

# Load data

filename = args.filename
# filename = 'data/reads/split/reads_150_aug_mut.fa'
read_headers, reads = parse_fasta(filename)

# Sample reads
sample_mask = np.random.rand(len(reads)) < args.aug_frac
# sample_idx = np.random.choice(len(reads), args.num_reads,
#                             replace=False)
sample_reads = reads[sample_mask]
sample_headers = read_headers[sample_mask]

# Save sampled data
ext = os.path.splitext(args.out_path)[1]  # File extension should be FASTA
if ext in ['.fa', '.fasta']:
    out_path = args.out_path
else:
    out_path = args.out_path + '.fa'

with open(out_path, 'w') as wfile:
    print('Creating file: {}'.format(out_path))
    for header, read in zip(sample_headers, sample_reads):
        wfile.write('>' + header + '\n')
        wfile.write(read + '\n')

