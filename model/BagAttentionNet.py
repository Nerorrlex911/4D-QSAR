import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.nn import Linear, ReLU, Sequential, Sigmoid
from typing import Sequence, Tuple

class MainNet:
    """
    Abstract class not intended to be invoked directly.
    Main net for multiple-instance learning via convolution on conformers. (conformation ensembles are hereafter termed "bags".)
    Linear transform maps each conformer to new hidden space of dimensionality hd (hd1,hd2,hd3..) This is perfomed seqentially ndim-1 times,
    with RelU nonlinearity after each linear layer (linear layer can be viewed as a set
    of 1-dimensional convolution filters). The result is learnt representation of bags of shape Nmols*Nconf*Nhlast, where
    Nmols - number of molecules, Nconf -number of conformers and Nhlast - dimensionality of last hidden layer.

    This learnt representation (as opposed to original one) is assumed to help better predict property studied.
    THe same net can be used for single instance learning, in which case it essentially is MLP.
    """

    def __new__(cls, ndim: Sequence):
        """
        Parameters
        -----------
        ndim: Sequence
        Each entry of sequence Specifies the number of nodes in each layer and length
        of the sequence specifies number of layers
        Returns
        --------
        MainNet instance
        """
        ind, hd1, hd2, hd3 = ndim
        net = Sequential(Linear(ind, hd1),
                         ReLU(),
                         Linear(hd1, hd2),
                         ReLU(),
                         Linear(hd2, hd3),
                         ReLU())

        return net
    
class Detector(nn.Module):
    def __new__(cls, input_dim:int, det_dim: Sequence):
        input_dim = input_dim
        attention = []
        for dim in det_dim[1:]:
            attention.append(Linear(input_dim, dim))
            attention.append(Sigmoid())
            input_dim = dim
        attention.append(Linear(input_dim, 1))
        net = Sequential(*attention)
        return net
    
class Estimator(nn.Module):
    def __new__(cls, input_dim:int):
        net = Linear(input_dim, 1)
        return net


    
class BagAttentionNet(nn.Module):
    def __init__(self, ndim: Sequence, det_ndim: Sequence, init_cuda: bool = False):
        """
              Parameters
              ----------
              ndim: Sequence
              Hyperparameter for MainNet: each entry of sequence specifies the number of nodes in each layer and length
              of the sequence specifies number of layers
              det_ndim: Sequence
              Hyperparameter for attention subnet: each entry of sequence specifies the number of nodes in each layer and length
              of the sequence specifies number of layers
              init_cuda: bool, default is False
              Use Cuda GPU or not?

              """
        super().__init__()
        input_dim = ndim[-1]
        self.main_net = MainNet(ndim)
        self.estimator = Estimator(input_dim)
        self.detector = Detector(input_dim, det_ndim)

        if init_cuda:
            self.main_net.cuda()
            self.detector.cuda()
            self.estimator.cuda()
    def forward(self, bags):
        # bags: Nmol*Nconf*Ndesc
        bags = self.bags(bags)
        # bags: Nconf*Nmol*Ndesc
        bags = bags.permute(1, 0, 2)
        # bags: Nconf*Nmol*Nhid
        bags, _ = self.attention(bags, bags, bags)
        # bags: Nmol*Nconf*Nhid
        bags = bags.permute(1, 0, 2)
        # bags: Nmol*Nhid
        bags = bags.mean(dim=1)
        # bags: Nmol*Nclass
        bags = self.fc(bags)
        return bags
