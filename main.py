
#!/usr/bin/env python3.8
'''
Written by: Saksham Consul 04/28/2023
Cleaned up main.py
'''
import os
import time
import numpy as np
import torch
import logging

from utils.general_tools import get_args, backup_files
# from utils.plotting import simple_plot, plot_labels
from utils.train_tools import get_train_valid_test_data
from utils.dataset import kmerDataset

from model import XVir
from trainer import Trainer
import pdb


class MyFilter(object):
    def __init__(self, level):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno <= self.__level


def main(args):
    time_string = args.experiment_name + '-' + \
        time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

    # Load data
    dataset = kmerDataset(args)
    train_dataset, valid_dataset, test_dataset, = get_train_valid_test_data(dataset, args)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    model = XVir(
        args.read_len, args.ngram, args.model_dim, args.num_layers, args.dropout
    )

    # Load the trained model, if needed
    if args.load_model or args.eval_only:
        print("Loading model from", args.model_path)
        try:
            model.load_state_dict(torch.load(args.model_path))
        except FileNotFoundError:
            print("Model not found, exiting")
            exit(1)
    # Create create tensorboard logs
    if args.eval_only:
        log_writer_path = './logs/eval'
    else:
        log_writer_path = './logs/runs'
    if not os.path.exists(log_writer_path):
        os.makedirs(log_writer_path)

    # Set up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(os.path.join(log_writer_path, 'log-' + time_string))
    c_handler.setLevel(logging.WARNING)
    f_handler.setLevel(logging.INFO)
    f_handler.addFilter(MyFilter(logging.INFO))

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    # Trainer
    trainer = Trainer(model, logger, time_string, args)

    if args.eval_only is False:
        # Backup all py files[great SWE practice]
        backup_files(time_string, args, None)
        # Train
        trainer.train(train_loader, valid_loader, args)

    # Test
    print("Test accuracy", trainer.accuracy(test_loader))
    logger.info("Test accuracy: {:.3f}".format(
        self.accuracy(test_loader)))

    # Visualize outputs
    # trainer.eval_output(train_loader, 'train')
    # trainer.eval_output(valid_loader, 'val')
    # trainer.eval_output(test_loader, 'test')


if __name__ == "__main__":
    args = get_args()
    main(args)
