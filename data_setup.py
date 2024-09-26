
"""
Contain functionalities for creating testing and training dataloader.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple


def create_datasets(train_transform: transforms, test_transform: transforms, train_dir:str, test_dir:str):
    """
    Create a training and testing dataset from files.
    Args:
    train_transform: the transform applied to train data.
    test_transform: the transform applied to test data.
    train_dir: training directory.
    test_dir: testing directory
    Return:
    train_data, test_data
    """

    train_data = datasets.ImageFolder(
        root = train_dir,  #the training directory
        transform = train_transform)
    
    test_data = datasets.ImageFolder(
        root = test_dir,
        transform = test_transform)

def create_dataloader(
    *data_and_transforms: Tuple[Dataset, bool],
    batch_size: int,
    num_workers: int = os.cpu_count(),
):
    """Create training and testing dataloaders
       Takes a testing and training directory and turn them into Pytorch Datasets and Dataloaders.

       Args:
       batch_size: Number of samples per batch in each of the DataLoaders.
       num_workers: An integer for number of workers per DataLoader.
       *data_and_transforms: lists of dataset and whether to shuffle dataset. 
       All datasets shoule have the same classes (targets)

       Returns:
       A list of *dataloaders of input Datasets, organized by input order, and class_names.
       where class_names is a list of the target classes.
    """
    dataloaders = []
    for dataset, shuffle in data_and_transforms:
    
        dataloader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = True,
        shuffle = shuffle,
        )

        dataloaders.append(dataloader)

    class_names = data_and_transforms[0][0].classes #get the classes

    return dataloaders, class_names
