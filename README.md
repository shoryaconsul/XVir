# XVir: A Transformer-Based Architecture for Identifying Viral Reads from Cancer Samples
**Authors:** Shorya Consul, John Robertson, Haris Vikalo

## Requirements

The YAML file `environment.yml` specifies the dependencies reqired to run XVir and competing benchmarks ([DeepVirFiner](https://github.com/jessieren/DeepVirFinder) and [Virtifier](https://github.com/crazyinter/Seq2Vec)). I've provided one example file. To create the environment run `conda env create -f environment.yml`

### File Structure
```
- utils/
  - __init__.py             : Needed for python packaging for `utils`
  - dataset.py              : Script to create dataset for XVir model
  - collate_data.py         : Script for creating Pickle data objects from .txt files
                              for numerically encoded reads
  - general_tools.py        : General tools for arguments, and backup while training
  - train_tools.py          : Tools for training
  - sample_data.py          : Subsampling reads from given read FASTA file or Pickle object
  - train_test_val_split.py : Script to take input read FASTA files and write output splits
                              into individual FASTA files
  - visualize_data.py       : Script to create t-SNE and MDS visualizations of reads
  - fastq2fasta.sh          : Bash script to create FASTA files corresponding to input FASTQ files
  

- data/                 : Place to store data. Additional documentation can be found in the README
                          included in the _data_ directory.
- logs/                 : Place where logs and model weights will be saved. 
                          The model weights for the 150bp model have been included. 
- model.py              : XVir model specification
- main.py               : Main script
- trainer.py            : Script for trainer. Invoked whenever training XVir.
- __init__.py           : Needed for python packaging
- environment.yml       : Dependencies for environment

- setup.sh              : Script to set up environment variables
- visualize_results.py  : Visualize results of chosen model

- LICENSE
- README.md
```
## To Run

Set up the required environment variables by runing `source setup.sh`.

### Inference Only
To use a trained XVir model for inference, we've included an `inference.py` script
We've provided the model weights for the base 150bp model in the /logs/ folder. Given a fasta file with 150bp reads, you may call it as:

`python inference.py --model_path=./logs/XVir_150bp_model.pt --input=./path/to/fasta.fa`

This will create a fasta.fa.output.txt in the same location as the input, containing the name of each read along with the probability that the read is HPV positive.
For other models, you can also specify the flags `--read_len`, `--ngram`, `--model_dim`, and `--num_layers` (as in `main.py`)
Inference batch size can be changed from the default (100) with `--batch_size` and GPU can be enabled by passing `--cuda`

### Training XVir on User Data (Recommended)
The script `main.py` is the primary entry point for the XVir pipeline. It includes the functionality for training, testing, and validating an XVir model on custom data.

`python main.py <args>`

For example, when specifying training, test and validation sets, XVir can be trained by running
`python main.py -s --train-data-file train_data.pkl --val-data-file val_data.pkl --test-data-file test_data.pkl --data-path data/ --device cuda`

The full list of arguments can be found in `utils/general_tools.py`, along with the default values for creating our XVir model. To prepare your data for training, please see the tools we've provided in the `data` folder.