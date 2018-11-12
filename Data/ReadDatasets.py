import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import pandas as pd
import numpy as np
from sklearn import preprocessing
import argparse

def Parser(argparse):

        parser = argparse.ArgumentParser(description='VAE MNIST Example')
        parser.add_argument('--hidden-layers', type=list, default=[200,100], metavar='N',
                    help='how many layers will the encoder have')
        parser.add_argument('--encoding-dim', type=int, default=10, metavar='N',
                    help='encoding dimension')
        parser.add_argument('--learning-rate', type=int, default=0.001, metavar='N',
                    help='how many layers will the encoder have')
        parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 128)')
        parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
        args = parser.parse_args(args=[])


        args.cuda = not args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(args.seed)

        device = torch.device("cuda" if args.cuda else "cpu")

        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        
        return kwargs, device, args
    

    

def scale_data(data):
    return preprocessing.MinMaxScaler().fit_transform(data)

class ReadDataset_train_random(Dataset):
   #""" Diabetes dataset."""

   # Initialize your data, download, etc.
    def __init__(self):
        x = np.loadtxt('random.csv',
                       delimiter=',', dtype=np.float32)
        y = np.loadtxt('random.csv',
                       delimiter=',', dtype=np.float32)        
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

class ReadDataset_test_random(Dataset):
   #""" Diabetes dataset."""

   # Initialize your data, download, etc.
    def __init__(self):
        x = np.loadtxt('random_test.csv',
                       delimiter=',', dtype=np.float32)
        y = np.loadtxt('random_test.csv',
                       delimiter=',', dtype=np.float32)        
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len  
    
class ReadDataset_train_Sanger(Dataset):
   #""" Diabetes dataset."""

   # Initialize your data, download, etc.
    
    def __init__(self):
        x = np.loadtxt('./Data/Zebrafish/GE_mvg.csv',
                       delimiter=',', dtype=np.float32)
        y = np.loadtxt('./Data/Zebrafish/GE_mvg.csv',
                       delimiter=',', dtype=np.float32)
        x=scale_data(x)
        y=scale_data(y)
        x=x[:int(x.shape[1]*(1-0.2)),:]
        y=y[:int(y.shape[1]*(1-0.2)),:]
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len    

class ReadDataset_test_Sanger(Dataset):
   #""" Diabetes dataset."""

   # Initialize your data, download, etc.
    def __init__(self):
        x = np.loadtxt('./Data/Zebrafish/GE_mvg.csv',
                       delimiter=',', dtype=np.float32)
        y = np.loadtxt('./Data/Zebrafish/GE_mvg.csv',
                       delimiter=',', dtype=np.float32)
        #x=x.T
        #y=y.T
        x=scale_data(x)
        y=scale_data(y)
        x=x[int(x.shape[1]*(0.2)):,:]
        y=y[int(y.shape[1]*(0.2)):,:]
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len        