import torch
from torch.utils.data import random_split
from torch.nn import functional as F
from tqdm import tqdm


def get_data_label_info(dataset):
    # Get information of how many class per sample
    labels = []
    for i in tqdm(range(len(dataset))):
        # if (i+1)%int(1e4) == 0:
        #     print('%.1f %% of data parsed' %((i+1)/len(dataset)*100))
        _, label = dataset[i]  # (num_sequence, n_classes)
        labels.append(label)
    labels = torch.stack(labels)

    return labels


def get_train_valid_test_data(dataset, args):
    train_size = int(args.train_split * len(dataset))
    valid_size = int(args.valid_split * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, [train_size, valid_size, test_size])

    print("Train size: ", train_size)
    print("Valid size: ", valid_size)
    print("Test size: ", test_size)
    torch.set_printoptions(sci_mode=False, precision=3)
    # print("Train label info:", torch.sum(
    #     get_data_label_info(train_dataset), axis=0).item()/len(train_dataset))
    # print("Valid label info:", torch.sum(
    #     get_data_label_info(valid_dataset), axis=0).item()/len(valid_dataset))
    # print("Test label info:", torch.sum(
    #     get_data_label_info(test_dataset), axis=0).item()/len(test_dataset))

    # n_classes = 2
    # weights = 1/(n_classes - 1)*torch.ones(n_classes)
    # weights[-1] = 0
    # if args.use_class_weights:
    #     print("Computing custom class weights")
    #     # Calculating class weights based on Class-Balanced Loss Based on Effective Number of Samples
    #     # Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, Serge Belongie
    #     num_per_class = torch.mean(
    #         torch.sum(get_data_label_info(train_dataset), axis=0), axis=0)[:-1]

        # effective_num = 1.0 - torch.pow(args.class_weight, num_per_class)
        # weights = (1.0 - args.class_weight) / effective_num
        # weights = torch.sum(torch.pow(num_per_class, 2)) / \
        #     (torch.pow(num_per_class, 2))
        # zero_tensor = torch.tensor([0])
        # weights = weights / torch.sum(weights)
        # weights = torch.cat((weights, zero_tensor), dim=0)

    # print("Effective weight for each claass: ", weights)
    return train_dataset, valid_dataset, test_dataset
