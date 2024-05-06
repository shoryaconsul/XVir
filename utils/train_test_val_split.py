# This Python file is used to take input FASTA files and splits them into training, validation and test sets.
# The reads are then saved as individual FASTA files.

import numpy as np
import os
import argparse


def split_data(headers, reads, outpath, split_ratio=[0.8, 0.1, 0.1]):
    """
    Split data into training, validation and test sets and write them to separate files.

    Parameters
    ----------
    headers : list
        List of headers of the reads
    reads : list
        List of read sequences
    outpath : string
        Path to output directory
    split_ratio : list (3-tuple)
        Ratio of training, validation and test data

    Returns
    -------
        None
    """

    indices = np.arange(len(headers))
    randnum = np.random.rand(len(headers))
    train_indices = indices[randnum < split_ratio[0]]
    val_indices = indices[(randnum >= split_ratio[0]) & (randnum < split_ratio[0] + split_ratio[1])]
    test_indices = indices[randnum >= split_ratio[0] + split_ratio[1]]

    with open(os.path.join(outpath, 'train.fa'), 'w') as f:
        for i in train_indices:
            f.write(headers[i] + '\n')
            f.write(reads[i] + '\n')
    
    with open(os.path.join(outpath, 'val.fa'), 'w') as f:
        for i in val_indices:
            f.write(headers[i] + '\n')
            f.write(reads[i] + '\n')

    with open(os.path.join(outpath, 'test.fa'), 'w') as f:
        for i in test_indices:
            f.write(headers[i] + '\n')
            f.write(reads[i] + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--viral', type=str, help="Path to viral FASTA file")
    parser.add_argument('-n', '--nonviral', type=str, help="Path to nonviral FASTA file")
    parser.add_argument('-o', '--output', type=str,
                    help="Output directory")  # Directory to store training, validation and test sets
    args = parser.parse_args()

    headers = []
    reads = []
    for file in [args.viral, args.nonviral]:
        if not os.path.exists(file):
            print('File %s does not exist' %file)
            exit(1)
    
    with open(args.viral, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                headers.append(">VIR_" + line[1:])
            else:
                reads.append(line)
        
    with open(args.nonviral, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                headers.append(">HUM_" + line[1:])
            else:
                reads.append(line)

    split_data(headers, reads, args.output)