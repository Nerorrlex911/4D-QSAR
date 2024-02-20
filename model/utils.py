import torch
from typing import Tuple
from data.dataset import MolDataSet

def data_split(bags: torch.Tensor, labels: torch.Tensor, split: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split data into train and test set.
    Parameters
    ----------
    bags: torch.Tensor
    labels: torch.Tensor
    split: float, default is 0.8
    Returns
    --------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    """
    n = bags.shape[0]
    idx = torch.randperm(n)
    n_train = int(n * split)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    return bags[train_idx], labels[train_idx], bags[test_idx], labels[test_idx]

def train_test_val_split(bags: torch.Tensor, labels: torch.Tensor, train: float = 0.7, test: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split data into train ,val and test set.
    Parameters
    ----------
    bags: torch.Tensor
    labels: torch.Tensor
    train: float, default is 0.8
    test: float, default is 0.8
    Returns
    --------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    """
    n = bags.shape[0]
    idx = torch.randperm(n)
    n_train = int(n * train)
    n_test = int(n * test)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:n_train + n_test]
    val_idx = idx[n_train + n_test:]
    return bags[train_idx], labels[train_idx], bags[test_idx], labels[test_idx], bags[val_idx], labels[val_idx]

def train_test_val_split(dataset:MolDataSet, train: float = 0.7, test: float = 0.2) -> Tuple[MolDataSet, MolDataSet, MolDataSet]:
    """
    Split data into train ,val and test set.
    Parameters
    ----------
    dataset: MolDataSet
    train: float, default is 0.8
    test: float, default is 0.8
    Returns
    --------
    Tuple[MolDataSet, MolDataSet, MolDataSet]
    """
    n = len(dataset)
    idx = torch.randperm(n)
    n_train = int(n * train)
    n_test = int(n * test)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:n_train + n_test]
    val_idx = idx[n_train + n_test:]
    return MolDataSet([dataset[i] for i in train_idx]), MolDataSet([dataset[i] for i in test_idx]), MolDataSet([dataset[i] for i in val_idx])