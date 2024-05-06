# Introduce mutations into read data

import numpy as np
from numpy import random as rn
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('filepath', type=str,
                    help='The path to load data')
parser.add_argument('outpath', type=str,
                    help='The path to load data')
parser.add_argument('-l', '--read-len', type=int, default=150,
                    help='Read length to output, i.e., read length')
parser.add_argument('-s', '--sub-rate', type=float, default=0.1,
                    help='The substitution rate')
parser.add_argument('-i', '--ins-rate', type=float, default=None,
                    help='The insertion rate')
parser.add_argument('-d', '--del-rate', type=float, default=None,
                    help='The deletion rate')
parser.add_argument('--aug', action="store_true",
                    help="Generate data for augmenting training set"
                    )  # Use when generating data for augmenting training set

def mutate_read(read, read_len, sub_rate, ins_rate, del_rate):

    if len(read) < read_len:
        raise ValueError("Read length is longer than input reads.")
    # elif len(read) == read_len and del_rate is not None:
    #     raise ValueError("Read length is equal to input reads, cannot delete.")

    bases = ['A', 'C', 'G', 'T']

    while True:
        is_mutated = False
        read_arr = np.array(list(read))
        if sub_rate is not None:  # Substitutions
            sub_mask = rn.rand(len(read_arr)) < sub_rate  # True for subsitutions
            subs = rn.choice(bases, size=len(read_arr), replace=True)
            read_arr[sub_mask] = subs[sub_mask]

        if del_rate is not None:  # Deletions
            del_mask = rn.rand(len(read_arr)) < del_rate
            read_arr = read_arr[~del_mask]

        if ins_rate is not None:  # Insertions
            ins_mask = rn.rand(len(read_arr)) < ins_rate
            ins = rn.choice(bases, size=len(read_arr), replace=True)
            ins_idx = np.nonzero(ins_mask)[0]  # indices of insertions
            read_arr = np.insert(read_arr, ins_idx, ins[ins_mask])

        if len(read_arr) < read_len:  # Regenerate mutations if read is too short
            continue
        else:  # Mutations complete
            break
    
    mutated_read = ''.join(read_arr)
    if mutated_read not in read:
        is_mutated = True
        
    start_idx = rn.randint(len(mutated_read) - read_len + 1)  # Randomly select read
    return is_mutated, mutated_read[start_idx:start_idx+read_len]
    

if __name__ == "__main__":
    args = parser.parse_args()

    out = open(args.outpath, 'w')
    with open(args.filepath, 'r') as f:
        for lines in tqdm(f):
            if lines[0] == '>':
                header = lines.strip()
            else:
                read = lines.strip()
                if 'HPV' in header or 'VIR' in header:  # Mutate only viral reads
                    is_mutated = False
                    while not is_mutated:
                        is_mutated, mutated_read = mutate_read(
                            read, args.read_len, args.sub_rate, args.ins_rate, args.del_rate
                            )
                elif not args.aug:  # Do not mutate non-viral reads
                    start_idx = rn.randint(len(read) - args.read_len + 1)  # Randomly select read
                    mutated_read = read[start_idx:start_idx + args.read_len]
                else:  # Skip non-viral reads when augmenting training set
                    continue

                out.write(header + '\n')
                out.write(mutated_read + '\n')
    out.close()
