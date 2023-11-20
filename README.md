<<<<<<< HEAD
# XVir (README IS WIP)
Transformer-based classifier for viral read identification in cancerous samples

## Requirements

Feel free to update your `environment.yml` file. I've provided one example file. To create the environment run `conda env create -f environment.yml`. Activate with `conda activate XVir` before running `main.py`.

```Note: This environment was designed to run XVir with PytorchROCM 5.2 and might need adjustment for other (ie Cuda/CPU) Systems```

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

`python main.py <args>`
=======
## Command-Line Arguments

| Argument    | Description | Syntax | Options (Default)* | 
| :---        |    :----:   |          :----: | :----: |
|   --data-path    |   The path to load data     |      | str ('data')* |
|  --data-file  |     The path to load data    |         |  str ('proc_data')*|
|  --train-data-file   |  The path to load train data |        | str ('split/train_data.pkl')* |
|  --val-data-file       |    The path to load validation data   |            | str ('split/val_data.pkl')* |
|  --test-data-file     |    The path to load test data    |     | str ('split/test_data.pkl')*  |
|  --train-split  |    Train split     |         | float < 1.0; (0.8)*  |
|--valid-split |     Validation split    |         | float < 1 - train-split; (0.1)* |
|--experiment-name |     Name of the experiment    |         | str ('XVir')* |
|--device |     What to use for compute [GPU, CPU] will be called.    |         | 'cuda'*, 'cuda:[int]', 'cpu' |
|--seed |    Random seed     |         | int (4)* |
|--num-processes |    The number of parallel processes used for training     |         | int (1)* |
|--read_len |     The input dimension, i.e., read length    |         | int (150)* |
|--ngram |      Length of N-gram   |       | int (6)*  |
|--model-dim |    The embedding dimension of transformer     |         | int (128)* |
|--num-layers |     The number of layers    |         | int (1)* |
|--batch-size |    The batch size     |         | int (100)* |
|--dropout |     The dropout rate    |         | float < 1.0 (0.1)* |
|--mask-rate |     The masking rate    |         | float < 1.0; (None)* |
|--n-epochs |     The number of epochs    |         | int (25)*|
|--learning-rate |     The learning rate    |         | float (1e-3)* |
|--weight-decay |     The weight decay rate    |         | float (1e-6) |
|--eval-only |     Only Evaluate the model    |   Append without argument(s)    | Disabled* |
|-s |     Passing splits for training and testing    |     Append without argument(s)    | Disabled* |
|--load-model |     Load Model    |         | bool (False)* |
|--model-path |     The path to load model    |         | str ('logs/experiment/XVir.pt')* |
|--model-save-interval |     How often to save the model    |         | int (5)*|
|--model-update-interval |     How often to update the model    |         | int (2)* |
|--model-save-path |     The path to save the trained model    |         | str ('./logs/experiment/XVir_models')*  |
|--print-log-interval |     How often to print training logs    |         | int (1)* |
|--val-log-interval |    How often to print validation logs     |         | int (5)* |

## Example Usage
