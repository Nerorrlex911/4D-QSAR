import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

class BagAttentionNet(nn.Module):
    def __init__(self, nconf, ndesc, nclass, nhead=8, nhid=128, dropout=0.1):
        super(BagAttentionNet, self).__init__()
        self.nconf = nconf
        self.ndesc = ndesc
        self.nclass = nclass
        self.nhead = nhead
        self.nhid = nhid
        self.dropout = dropout
        self.bags = nn.Sequential(
            nn.Linear(ndesc, nhid),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attention = nn.MultiheadAttention(nhid, nhead, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(nhid, nclass),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
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
