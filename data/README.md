This folder contains scripts required to simulate the data generation. These scripts were used to generate the semi-experimental data used to benchmark XVir.

# Commands
- **Generate mutated viral reads from a given set of reads:** The output read length has to be less than the length of the input reads. The output FASTA file will contain mutated viral reads (header contains 'HPV') while leaving the remaining reads unchanged. 
  > python mutate_reads.py [input FASTA file] [output FASTA file] -l [Length of output reads] -s [sub rate] -i [ins rate] -d [del rate]

- **Augment training set with mutated viral reads:**
  > bash gen_aug_data.sh
  
  This Shell script invokes the following scripts:
  - **mutate_reads.py:** Mutate viral reads to augment training set.
  - **sample_fasta.py:** Sample reads from the mutated reads. The option _[-a, --aug-frac]_ should be set to **half** the required fraction.
  - **pickle_fasta.py:** Pickles the FASTA file into a dictionary containing the reads and their respective labels for consumption by the XVir pipeline.

