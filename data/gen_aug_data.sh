#!/bin/bash

INPUT_FILE="reads/split/reads_250_train.fa"
OUTLEN=250
OUTPRE="reads/split/reads_"

# Generate mutated reads
python mutate_reads.py $INPUT_FILE $OUTPRE$OUTLEN"_aug_mut_5.fa" -l $OUTLEN -s 0.0375 -i 0.00625 -d 0.00625 --aug
python mutate_reads.py $INPUT_FILE $OUTPRE$OUTLEN"_aug_mut_10.fa" -l $OUTLEN -s 0.075 -i 0.0125 -d 0.0125 --aug
python mutate_reads.py $INPUT_FILE $OUTPRE$OUTLEN"_aug_mut_15.fa" -l $OUTLEN -s 0.1125 -i 0.018725 -d 0.01875 --aug
python mutate_reads.py $INPUT_FILE $OUTPRE$OUTLEN"_aug_mut_20.fa" -l $OUTLEN -s 0.15 -i 0.025 -d 0.025 --aug

# Cocatenate all the generated files
# ls $INPUT_FILE $OUTPRE$OUTLEN"_aug_mut_"*".fa"
cat $OUTPRE$OUTLEN"_aug_mut_"*".fa" > $OUTPRE$OUTLEN"_aug_mut.fa"

# Sample reads
python sample_fasta.py $OUTPRE$OUTLEN"_aug_mut.fa" -o $OUTPRE$OUTLEN"_aug_samp_0.1.fa" -a 0.05  
cat "reads/split/reads_"$OUTLEN"_train.fa" $OUTPRE$OUTLEN"_aug_samp_0.1.fa" >  "reads/split/reads_"$OUTLEN"_train_aug_0.1.fa"

# Pickle data for use by XVir
python pickle_fasta.py -l $OUTLEN -f "reads/split/reads_"$OUTLEN"_train_aug_0.1.fa" -o reads/split/train_data_"$OUTLEN"bp_aug_0.1.pkl
