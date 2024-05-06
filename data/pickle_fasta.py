# Parse FASTA file to generate read-matrix + label array

import os
import numpy as np
import argparse
from tqdm import tqdm
import bz2
import _pickle as cPickle


def parse_fasta(args):
    base_int = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    with open(args.file, 'r') as f:
        read_list = []
        y_list = []
        y = 0
        for line in f:
            line = line.strip()
            if line.startswith('>'):  # Header
                if 'HPV' in line or 'VIR' in line:  # Viral read
                    y = 1
                else:  # Non-viral read
                    y = 0
                head = line
            else:  # Read sequence
                if(len(line) != args.length):  # Skip reads that are not of specified length
                    print('Read length not equal to specified length')
                    continue
                else:
                    read_list.append(np.array([base_int[base] for base in line]))
                    y_list.append(y)
    reads = np.array(read_list)
    read_labels = np.array(y_list)

    print('Number of reads: %d' %(len(reads)))
    classes, class_cnt = np.unique(read_labels, return_counts=True)
    print('Number of non-viral reads: %d, viral reads: %d' %(class_cnt[0], class_cnt[1]))

    # Save data to pickle file
    data_dict = {'reads': reads, 'labels': read_labels}
    _, output_ext = os.path.splitext(args.output)  # Check that the output file is .pkl
    if output_ext == '.pkl':
        pfile = args.output
    else:
        pfile = args.output + '.pkl'
    with bz2.BZ2File(pfile, 'wb') as pfile:
        cPickle.dump(data_dict, pfile, protocol=4)
    
    print('Data processing completed ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--length', type=int, default=150,
                    help="Length of reads")
    parser.add_argument('-f', '--file', type=str, default=None,
                    help="File to collate data from")  # Use when only one file is needs to be processed
    parser.add_argument('-o', '--output', type=str,
                    help="Output file path")  # Only used when -f is specified
    args = parser.parse_args()

    parse_fasta(args)