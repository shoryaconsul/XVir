import os
import time
import numpy as np
import pandas as pd
import random
from sklearn.metrics import roc_auc_score, roc_curve

import torch
import torch.nn as nn

from tqdm import tqdm
import logging
from matplotlib import pyplot as plt


class Trainer(object):
    def __init__(self, model, writer, timeStr, args):
        self.args = args
        self.model = model
        self.writer = writer
        if torch.cuda.is_available():
            if 'cuda' in args.device:
                device = torch.device(args.device)
                print('---------Using GPU-------------------')
            elif args.device == 'cpu':
                device = torch.device('cpu')
                print('---------Using CPU-------------------')
            else:
                device = torch.device('cuda')
                print('---------Using GPU-------------------')
        else:
            device = torch.device('cpu')
            print('---------Using CPU-------------------')
        self.device = device
        self.model.to(device)
        if device == torch.device('cuda'):
            assert next(self.model.parameters()
                        ).is_cuda, 'Model is not on GPU!'

        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')
        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=args.learning_rate,
                                          weight_decay=args.weight_decay)
        self.timeStr = timeStr
        seed = args.seed

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

    def train(self, train_loader, val_loader, args):
        print("Starting training")
        batch_size = args.batch_size
        n_epochs = args.n_epochs
        model_save_path = os.path.join(args.model_save_path, self.timeStr)
        self.model.to(self.device)
        self.model.train()

        # Start training
        training_loss = []
        for epoch in tqdm(range(1, n_epochs + 1)):
            print('Epoch %d of %d' %(epoch, n_epochs))
            running_loss = 0.0
            batch_losses = []
            c = 0
            for x_batch, y_batch in train_loader:
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(x_batch.to(self.device), args.mask_rate)
                batch_size, _ = y_batch.shape
                loss = self.criterion(outputs, y_batch.to(self.device))
                if loss.isnan().any():
                    print("Loss is NaN!")
                    import pdb
                    pdb.set_trace()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                batch_losses.append(loss.item())
                c += 1
                if False and c % 20 == 19:    # print every 20 mini-batches
                    print(
                        f'[{epoch}, {c + 1:5d}] loss: {running_loss / 20:.3f}')
                    running_loss = 0.0
            training_loss.append(np.mean(batch_losses))

            # Print statistics & write logs.
            if epoch % args.print_log_interval == 0:
                print('[%d] training loss: %.3f' %
                      (epoch, np.mean(training_loss) / args.print_log_interval))
                self.writer.info("epoch : {}/{}, loss = {:.2f}".format(
                    epoch, n_epochs, np.mean(training_loss) / args.print_log_interval)
                    )
                training_loss = []

            if epoch % args.val_log_interval == 0:
                # Validation loss
                validation_loss = 0.0
                self.model.eval()
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        outputs = self.model(x_val.to(self.device))
                        batch_size, _ = y_val.shape
                        loss = self.criterion(outputs, y_val.to(self.device))
                        validation_loss += loss.item()

                print('[%d] validation loss: %.3f' %
                      (epoch, validation_loss / len(val_loader)))
                self.writer.info("epoch : {}/{}, val loss = {:.2f}".format(
                    epoch, n_epochs, validation_loss / len(val_loader)))
                self.writer.info("Accuracy: Training: {:.3f}, Validation: {:.3f}".format(
                    self.accuracy(train_loader), self.accuracy(val_loader))
                    )

                # self.writer.add_scalar('validation_loss',
                #                        validation_loss / len(val_loader),
                #                        epoch)
                # self.writer.add_scalar('validation_accuracy',
                #                        self.accuracy(val_loader),
                #                        epoch)

            # Save the trained  model
            if (epoch % args.model_save_interval == 0) and args.model_save_path != "":
                if epoch % args.model_update_interval == 0:
                    sub_time_str = time.strftime(
                        '%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                sub_time_str = time.strftime(
                    '%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
                torch.save(self.model.state_dict(), os.path.join(
                    model_save_path, self.timeStr + '_' + sub_time_str + ".pt"))

        print('Finished Training')


    def accuracy(self, loader, test=False):
        true_neg, true_pos = 0, 0
        total_neg, total_pos = 0, 0
        self.model.eval()
        prob = torch.empty(0)
        with torch.no_grad():
            for x_val, y_val in loader:
                outputs = self.model(x_val.to(self.device))
                predicted = (outputs.detach() > 0).float().to('cpu') 
                total_neg += y_val[y_val == 0].size(0)
                total_pos += y_val[y_val == 1].size(0)
                true_neg += (predicted[y_val == 0] == y_val[y_val == 0]).sum().item()
                true_pos += (predicted[y_val == 1] == y_val[y_val == 1]).sum().item()
                prob = torch.cat((prob, torch.sigmoid(predicted)), 0)
            # print('Correct: %d, Total %d' %(correct, total))
        prob_sd, prob_mean = torch.std_mean(prob)
        # print("Mean prediction probability: {:.3f}, Std: {:.3f}".format(prob_mean, prob_sd))
        
        if test:  # Compute TPR, TNR and PPV
            tpr = true_pos / total_pos
            tnr = true_neg / total_neg
            ppv = true_pos / (true_pos + (total_neg - true_neg))
            print("TPR: {:.3f}, TNR: {:.3f}, PPV: {:.3f}".format(tpr, tnr, ppv))
        return (true_neg + true_pos) / (total_neg + total_pos)


    def compute_roc(self, loader, save=True):
        self.model.eval()
        y_true = []
        y_prob = []
        with torch.no_grad():
            for x_val, y_val in loader:
                y_true.extend(y_val.numpy())  # Store true labels

                logits = self.model(x_val.to(self.device))
                pred = torch.sigmoid(logits.detach().to('cpu'))
                y_prob.extend(pred.numpy())

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)

        if save:
            target_directory = os.path.join(
                os.path.dirname(self.args.model_save_path), self.timeStr)
            zfile = os.path.join(target_directory, 'roc_res.npz')
            np.savez(zfile, fpr=fpr, tpr=tpr)

            fig_file = os.path.join(target_directory, 'roc.png')
            plt.figure(figsize=(10, 10))
            plt.plot(fpr, tpr, color='darkorange')
            plt.savefig(fig_file)

        return roc_auc


    def accuracy_per_class(self, loader, num_class):
        match_cnt = np.zeros(num_class)
        total_cnt = np.zeros(num_class)

        self.model.eval()
        prob = torch.empty(0)
        with torch.no_grad():
            for x_val, y_val in loader:
                outputs = self.model(x_val.to(self.device))
                predicted = (outputs.detach() > 0).float().to('cpu')
                for i in range(num_class):
                    match_cnt[i] += (predicted[y_val == i] == (y_val[y_val == i] > 0).float()).sum().item()
                    total_cnt[i] += y_val[y_val == i].size(0)
        
        return match_cnt / total_cnt


    # def compute_roc_per_class(self, loader, num_class):
    #     self.model.eval()
    #     y_true = []
    #     y_prob = []
    #     with torch.no_grad():
    #         for x_val, y_val in loader:
    #             y_true.extend(y_val.numpy())  # Store true labels

    #             logits = self.model(x_val.to(self.device))
    #             pred = torch.sigmoid(logits.detach().to('cpu'))
    #             y_prob.extend(pred.numpy())

    #     auc_per_class = []
    #     y_true = np.array(y_true)
    #     y_prob = np.array(y_prob)
    #     for i in range(num_class):  # Compute AUROC for each class
    #         yi_true = (y_true[y_true == i] > 0).astype(int)
    #         yi_prob = y_prob[y_true == i]
    #         try:
    #             roc_auc = roc_auc_score(yi_true, yi_prob)
    #         except ValueError:
    #             auc_per_class.append(1.)

    #     return auc_per_class