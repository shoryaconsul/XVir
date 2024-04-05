# XVir
Transformer-based classifier for viral read identification in cancerous samples

## Requirements

Feel free to update your `environment.yml` file. I've provided one example file. To create the environment run `conda env create -f environment.yml`

### File Structure
```
- utils/
  - __init__.py         : Needed for python packaging for `utils`
  - dataset.py          : Script to create dataset
  - filtering.py        : Script for filtering. Needed for raw data processing
  - collate_data.py     : Script for collating raw data into structured data
  - general_tools.py    : General tools for arguments, and backup while training
  - specific_tools.py   : Specific tools for given application
  - plotting.py         : Some plotting scripts to plot results and data
  - train_tools.py      : Tools for training
- data/                 : Place to store data
- logs/                 : Place where logs will be saved
- model.py              : GRU Model
- main.py               : Main script
- trainer.py            : Script for trainer
- __init__.py           : Needed for python packaging
- environment.yml       : Dependencies for environment

- LICENSE
- README.md
```
## To Run

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

The full list of arguments can be found in `utils/general_tools.py`, along with the default values for creating our XVir model. To prepare your data for training, please see the tools we've provided in the `data/` folder.