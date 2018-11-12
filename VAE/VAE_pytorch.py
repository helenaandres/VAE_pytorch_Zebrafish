from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import pandas as pd



########### arguments parser ###########
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


class ReadDataset_train(Dataset):
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

class ReadDataset_test(Dataset):
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

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.input_linear=nn.Linear(100,20)
#        for i in H_l:
#            self.middle_linear=nn.Linear(i,i)
        self.enc_middle=nn.Linear(20,10)
        self.enc_1=nn.Linear(10,2)
        self.enc_2=nn.Linear(10,2)
        self.dec_0=nn.Linear(2,10)
        self.dec_middle=nn.Linear(10,20)
        self.output_linear=nn.Linear(20,100)

    def encode(self, x):
        h1 = F.relu(self.input_linear(x))
        h2 = F.relu(self.enc_middle(h1))
        return self.enc_1(h2), self.enc_2(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.dec_0(z))
        h4 = F.relu(self.dec_middle(h3))
        return torch.sigmoid(self.output_linear(h4))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 100))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class VAE_Sanger(nn.Module):
    def __init__(self):
        super(VAE_Sanger, self).__init__()
        self.input_linear=nn.Linear(1845,512)
        self.bn1=nn.BatchNorm1d(512)
#        for i in H_l:
#            self.middle_linear=nn.Linear(i,i)
        self.enc_middle=nn.Linear(512,128)
        #self.bn2=nn.BatchNorm1d(128)
        self.enc_1=nn.Linear(128,50)
        self.enc_2=nn.Linear(128,50)
        self.dec_0=nn.Linear(50,128)
        self.dec_middle=nn.Linear(128,512)
        #self.bn4=nn.BatchNorm1d(512)
        self.output_linear=nn.Linear(512,1845)
        
    def encode(self, x):
        h1 = F.relu(self.bn1(self.input_linear(x)))
        #h2 = F.relu(self.bn2(self.enc_middle(h1)))
        h2 = F.relu(self.enc_middle(h1))
        return self.enc_1(h2), self.enc_2(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        #h3 = F.relu(self.bn3(self.dec_0(z)))
        #h4 = F.relu(self.bn4(self.dec_middle(h3)))        
        h3 = F.relu(self.dec_0(z))
        h4 = F.relu(self.dec_middle(h3))
        return torch.sigmoid(self.output_linear(h4))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1,1845))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar    
    
    
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 1845), reduction='sum') # what if I just use x.view(-1, 100) ?

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #nll=(recon_x.view(-1, 1845)-x.view(-1, 1845)).pow(2).mean().cuda()
    #nll=(recon_x-x).pow(2).mean().cuda()

    return KLD + BCE 

def loss_function2(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 1845), reduction='sum') # what if I just use x.view(-1, 100) ?

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #nll=(recon_x-x).pow(2).mean().cuda()
    #nll=(recon_x.view(-1, 1845)-x.view(-1, 1845)).pow(2).mean().cuda()

    return 0.*KLD + BCE 

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

def compute_kernel(x,y):
    x_size=x.shape[0]
    y_size=y.shape[0]
    dim=x.shape[1]
    tiled_x=tile(x.view(x_size,1,dim),y.view(1,y_size,1))
    tiled_y=tile(x.view(y_size,1,dim),x.view(x_size,1,1))
    a=torch.exp(-torch.mean(torch.power(tiled_x-tiled_y,2),dim=2))/dim
    return a
    


def train(model, epoch, train_loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(model, epoch, test_loader, optimizer):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                #comparison = torch.cat([data[:n],
                #                      recon_batch.view(args.batch_size, 100)[:n]])
                #save_image(comparison.cpu(),
                #         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return mu, logvar
    
def train_warmup(model, epoch, train_loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function2(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test_warmup(model, epoch, test_loader, optimizer):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function2(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                #comparison = torch.cat([data[:n],
                #                      recon_batch.view(args.batch_size, 100)[:n]])
                #save_image(comparison.cpu(),
                #         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return mu, logvar