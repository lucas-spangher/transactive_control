import argparse
import torch
from lstm import LSTM
from train_and_test import Optimizer
from torch.utils.data import DataLoader
from train_and_test import Optimizer

import sys
sys.path.append("..")
from new_action_energy_data.dataset import Action_Energy_Dataset

def main(args):

    #Checks that paths exist before starting
    if(args.train_data_path == args.test_data_path):
        raise ValueError(('Train and Test Datasets must differ'))

    #Initializes trainining dataset
    train_dataset = Action_Energy_Dataset(args.train_data_path, 'train')
    
    #Checks that batch_size is realistic
    if(args.batch_size > len(train_dataset)):
        raise ValueError(("Batch Size {:d} cannot be greater than size of dataset {:d}".format(args.batch_sz, len(train_dataset))))
    
    #Initializes validation and "dummy test" dataset
    valid_dataset = Action_Energy_Dataset(args.train_data_path,'valid')
    test_dataset = Action_Energy_Dataset(args.test_data_path, 'dummy_test')

    batch_sz = args.batch_size

    #Creates dataloader for training, validation, "dummy test" datasets
    train_dataloader = DataLoader(train_dataset, batch_size= batch_sz,
                        shuffle=True, num_workers=1)
    valid_dataloader = DataLoader(valid_dataset, batch_size= batch_sz,
                        shuffle=True, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size= batch_sz,
                        shuffle=True, num_workers=1)


    # Init model and helper class
    model = LSTM(args.input_size, args.hidden_size, args.num_layers, batch_sz)
    helper = Optimizer(model, device = args.device)

    # Trains model
    helper.train(train_dataloader, valid_dataloader, args.num_epochs)

    # Gets test accuracy. (I used accuracy b/c our dummy test dataset is not a true test set)
    _, test_acc = helper.eval_(test_dataloader)

    print("Test Accuracy: {:f}".format((test_acc)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('LSTM Model')

    #Data
    parser.add_argument('--train_data_path', type=str, 
                    default="../new_action_energy_data/0_extra_train/linear_extratrain_0.csv",
                    help='Filepath of  training data csv.')

    parser.add_argument('--test_data_path', type=str, 
                    default="../new_action_energy_data/99_extra_train/linear_extratrain_99.csv",
                    help='Filepath of  dummy test data csv.')

    # Model
    parser.add_argument('--input_size', type=int, default=1,
                        help='Size of input at each time step')

    parser.add_argument('--hidden_size', type=int, default = 5, 
                        help = 'size of hidden dimension')

    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in model')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size')

    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')


    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main(args)