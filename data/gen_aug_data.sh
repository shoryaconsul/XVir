#!/bin/bash

# Generate 10% augmented data. To vary the fraction to x%, change the value of -a to x/2%.

# INPUT_FILE="reads/split/reads_250_train.fa"
# OUTPRE="reads/split/reads_"
# OUTLEN-150

INPUT_FILE="mixture/split/train.fa"
OUTPRE="mixture/split/train_"
OUTLEN=150

INPUT_PRE=${INPUT_FILE%.*}

# Generate mutated reads
python mutate_reads.py $INPUT_FILE $OUTPRE$OUTLEN"_aug_mut_5.fa" -l $OUTLEN -s 0.0375 -i 0.00625 -d 0.00625 --aug
python mutate_reads.py $INPUT_FILE $OUTPRE$OUTLEN"_aug_mut_10.fa" -l $OUTLEN -s 0.075 -i 0.0125 -d 0.0125 --aug
python mutate_reads.py $INPUT_FILE $OUTPRE$OUTLEN"_aug_mut_15.fa" -l $OUTLEN -s 0.1125 -i 0.018725 -d 0.01875 --aug
python mutate_reads.py $INPUT_FILE $OUTPRE$OUTLEN"_aug_mut_20.fa" -l $OUTLEN -s 0.15 -i 0.025 -d 0.025 --aug

# Cocatenate all the generated files
# ls $INPUT_FILE $OUTPRE$OUTLEN"_aug_mut_"*".fa"
cat $OUTPRE$OUTLEN"_aug_mut_"*".fa" > $OUTPRE$OUTLEN"_aug_mut.fa"

# Sample reads
python sample_fasta.py $OUTPRE$OUTLEN"_aug_mut.fa" -o $OUTPRE$OUTLEN"_aug_samp_0.1.fa" -a 0.05  # 0.05 = 10% augmentation
cat $INPUT_FILE $OUTPRE$OUTLEN"_aug_samp_0.1.fa" >  $INPUT_PRE"_aug_0.1.fa"

# Pickle data for use by XVir
# echo "{$(dirname $INPUT_FILE)"/train_data_"$OUTLEN"bp_aug_0.1.pkl"}"
python pickle_fasta.py -l $OUTLEN -f $INPUT_PRE"_aug_0.1.fa" -o $(dirname $INPUT_FILE)"/train_data_"$OUTLEN"bp_aug_0.1.pkl"
