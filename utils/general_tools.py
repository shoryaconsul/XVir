#!/usr/bin/env python3.8
'''
Written by: Saksham Consul 04/28/2023
Scripts needed for data handling, backing up, argument parsing
'''
import os
import argparse

from torch import save as torch_save
from shutil import copyfile, copytree


def get_args():
    parser = argparse.ArgumentParser(description='Arguments')
    # Data arguments
    parser.add_argument('--data-path', type=str, default='data',
                        help='The path to load data')
    parser.add_argument('--data-file', type=str, default='proc_data.pkl',
                        help='The path to load data')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Train split')
    parser.add_argument('--valid-split', type=float, default=0.1,
                        help='Valid split')
    # Experiment arguments
    parser.add_argument('--experiment-name', type=str,
                        default='XVir', help='Name of the experiment')
    # Basic arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='What to use for compute [GPU, CPU]  will be called')
    parser.add_argument('--seed',   type=int, default=4, help='Random seed')
    parser.add_argument('--num-processes', type=int, default=1,
                        help='The number of parallel processes used for training')
    # parser.add_argument('--plot', type=bool, default=False,
    #                     help='Plot the data')

    # Hyperparameters for Model
    parser.add_argument('--read_len', type=int, default=150,
                        help='The input dimension, i.e., read length')
    parser.add_argument('--ngram', type=int, default=3,
                        help='Length of N-gram')
    parser.add_argument('--model_dim', type=int, default=64,
                        help='The embedding dimension of transformer')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='The number of layers')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='The batch size')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='The dropout rate')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='The number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='The learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-6,
                        help='The weight decay rate')
    # parser.add_argument('--class-weight', type=float, default=0.9,
    #                     help='The weighting factor for calculating effetive class weight')
    # parser.add_argument('--use-class-weights', type=bool, default=False,
    #                     help='Use class weights')
    # Arguments for eval, logging, and printing
    parser.add_argument('--eval-only', type=bool, default=False,
                        help='Only Evaluate the model')
    parser.add_argument('--load-model', type=bool, default=False,
                        help='Load Model')
    parser.add_argument('--model-path', type=str, default='logs/experiment',
                        help='The path to load model')
    parser.add_argument('--model-save-interval',  type=int,
                        default=5, help='How often to save the model')
    parser.add_argument('--model-update-interval', type=int,
                        default=2, help='How often to update the model')
    parser.add_argument('--model-save-path', type=str,
                        default='./logs/experiment/XVir_results.pt',
                        help='The path to save the trained model')
    parser.add_argument('--print-log-interval', type=int,
                        default=1, help='How often to print training logs')
    parser.add_argument('--val-log-interval', type=int,
                        default=5, help='How often to print validation logs')

    args = parser.parse_args()

    return args


def store_args(args, target_directory):
    with open(os.path.join(target_directory, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))


def backup_files(time_string, args, policy=None):
    target_directory = os.path.join('./logs/experiment', time_string)

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    # Store the arguments
    store_args(args, target_directory)
    # copyfile('utils/dataset.py',    os.path.join(target_directory, 'dataset.py'))
    # copyfile('utils/tools.py',    os.path.join(target_directory, 'tools.py'))
    # copyfile('utils/train_tools.py',
    #          os.path.join(target_directory, 'train_tools.py'))
    # copyfile('main.py',    os.path.join(target_directory, 'main.py'))
    # copyfile('model.py',   os.path.join(target_directory, 'model.py'))
    # copyfile('trainer.py',   os.path.join(target_directory, 'trainer.py'))
    # environment_path = './'
    # environment_name = 'DL-Project'
    # environment_path = os.path.join(environment_path, environment_name)
    # try:
    #     copytree(environment_path, os.path.join(
    #         target_directory, environment_name))
    # except:
    #     pass
    try:
        torch_save(policy.state_dict(), os.path.join(
            args.model_save_path, + time_string + ".pt"))
    except:
        pass
