#!/usr/bin/env python3.8
'''
Written by: Saksham Consul 05/08/2023
Script to create dataset from the raw data
from datetime import datetime
'''
import os
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
import bz2
import pickle
# from utils.plotting import plot_graph


def main(args):
    # Set file paths
    raw_data_path = 'data/'
    viral_read_files = ['hpv_reads']
    nonviral_read_files = ['human_reads']
    output_pkl = 'data/proc_data.pkl'

    # Reading viral data files
    vread_list = []
    for vf in viral_read_files:
        vfile = os.path.join(raw_data_path, vf) + '.txt'
        vreads = np.loadtxt(vfile)
        if vreads.shape[1] != args.length:  # Check that reads are of correct length
            raise ValueError('Reads in %s are not %d bp.' %(vfile, args.length))
        vread_list.append(vreads)
    viral_reads = np.vstack(vread_list)
    viral_labels = np.ones(viral_reads.shape[0], dtype=np.int8)
    
    # Reading non-viral data files
    nvread_list = []
    for nvf in nonviral_read_files:
        nvfile = os.path.join(raw_data_path, nvf) + '.txt'
        nvreads = np.loadtxt(nvfile)
        if nvreads.shape[1] != args.length:  # Check that reads are of correct length
            raise ValueError('Reads in %s are not %d bp.' %(nvfile, args.length))
        nvread_list.append(nvreads)
    nonviral_reads = np.vstack(nvread_list)
    nonviral_labels = np.zeros(nonviral_reads.shape[0], dtype=np.int8)
    
    # Pickle data objects
    data_dict = {'reads': np.vstack((viral_reads, nonviral_reads)),
                'labels': np.concatenate((viral_labels, nonviral_labels))}
    with bz2.BZ2File(output_pkl,'wb') as pfile:
        pickle.dump(data_dict, pfile, protocol=4)
    
    print('Data processing completed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--length', type=int, default=150,
                    help="Length of reads")
    args = parser.parse_args()
    main(args)
