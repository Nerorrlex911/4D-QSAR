import torch
from typing import Tuple
from data.dataset import MolDataSet
from torch.utils.data import DataLoader, random_split, Dataset
import logging
from torch import Generator

def dataset_split(dataset: MolDataSet, train: float = 0.7, val: float = 0.2, generator: Generator = torch.Generator().manual_seed(42)) -> Tuple[Dataset,Dataset,Dataset]:
    total_len = len(dataset)
    train_len = int(total_len * train)
    test_len = int(total_len * val)
    val_len = total_len - train_len - test_len
    train_dataset,test_dataset,val_dataset = random_split(dataset=dataset,lengths=[train_len,test_len,val_len],generator=generator)
    logging.info(f'train_dataset:{len(train_dataset)} test_dataset:{len(test_dataset)} val_dataset:{len(val_dataset)}')
    return train_dataset,test_dataset,val_dataset

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