
"""
Contain various functions for Pytorch.
"""
import torch
from torch import nn
from torchvision import transforms
from pathlib import Path
from functools import partialmethod

def partial_class(cls, *args, **kwargs):
    class NewClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs) #assign innit of old class to new class

    return NewClass()

def create_writer(
    experiment_name: str,
    model_name: str,
    stats: str = None,
):

    """
    A function to create a torch.utils.tensorboard.SummeryWriter() instance.

    Args:
    experiment_name: the name of the experiment.
    model_name: the name of the model.
    stats: any other file name we want nested in log_dir, default to None

    Returns:
    SummaryWriter(log_dir = log_dir) where log_dir will be /runs/time/experiment_name/model_name/stats

    A string of the path name. (/runs/time/experiment_name/model_name/stats)
    
    """

    from datetime import datetime
    import os
    from torch.utils.tensorboard import SummaryWriter

    #get the timestamp of current date
    time = datetime.now().strftime('%d-%m-%Y') #get current date in dd/mm/yyyy format

    log_dir = os.path.join('runs', time, experiment_name, model_name)

    if stats != None:
        log_dir = os.path.join(log_dir, stats)

    print(f"Creating SummaryWriter, saving files to {log_dir}")
    
    return SummaryWriter(log_dir = log_dir)


def save_model(
  model: nn.Module,
  model_name: str,
  save_dir: str = 'data',
  full_model_save: bool = False,
):
    """
    A function for saving a Pytorch model.
    Args:
    model: A Pytorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
    """

    state_dict_path = Path(f"{save_dir}/{model_name}/{model_name}_state_dict.pth")
    print(f"Saving model's state dict to: {state_dict_path}")

    if not Path(f"{save_dir}/{model_name}/{model_name}_state_dict.pth").is_dir():
        Path(f"{save_dir}/{model_name}").mkdir(parents = True, exist_ok = True)

    torch.save(obj = model.state_dict(),
               f = state_dict_path)

    if full_model_save:
        full_model_path = Path(f"{save_dir}/{model_name}/{model_name}.pth")
        print(f"Saving model to: {full_model_path}")

        if not Path(f"{save_dir}/{model_name}/{model_name}.pth").is_dir():
            Path(f"{save_dir}/{model_name}").mkdir(parents = True, exist_ok = True)
        torch.save(obj = model,
                   f = full_model_path)


def save_results_json(file_name: str, results):
    """
    Save a file of the models results as a json file. 

    Args:
    file_name: the name of the file. 
    results: model's results, the content we want to save.
    """
    import json
    from pathlib import Path
    file_path = str(file_name)

    with open(file_path, 'w') as f:
        f.write(json.dumps(results))

def get_mean_and_standard_deviation(dataloader):
    """
    Get the mean and standard deviation of images in a Dataloader.

    Args:
    dataloader: the dataloader of the image dataset
    """
    mean = 0
    std = 0
    total_images = 0
    for images, _ in dataloader:
        batch_size = images.size(0) #get the size of the batches
        images = images.view(batch_size, images.size(1), -1) #faltten the image, except the color channels

        mean += images.mean(2).sum(0) #.sum(0) adds all the image together (add 32 images together in a batch of 32)

        std += images.std(2).sum(0)

        total_images += batch_size

    mean /= total_images
    std /= total_images

    return mean, std

def set_random_seed(seed:int = 314):
    """
    A function to set the random seed.

    Args:
    seed: the random seed to be set.
    """

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

class Transforms():
    def __init__(self):
        """
        A class that contain all the transforms that I use. 
        
        Return:
        A transforms.Compose. 

        """


    def transform_one(self, mean, std):
        """
        Args: 
        mean: the mean of the image set.
        std: the standard deviation of the image set. 
        """
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding = 4, padding_mode = 'reflect'),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace = True) #directly change without making copies means in_place
        ])

        return transforms
    def transform_two(self):
        """Return a transforms with:
        random crop, random horizontal flip, and to tensor. 
        used when you want a transform, but too lazy to use normalize 
        and still want sth halfway decent. 
        """
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding = 4, padding_mode = 'reflect'),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.ToTensor(),
        ])
        return transform


def get_model_summary(model, input_size, device: str = 'cpu'):
    """
    Print out the summary of the inbuilt model in torchvision. Need to import torchinfo first!
    Args:
    model: torch model
    input_size: size of input image
    device: the device the model is on. Default is 'cpu'
    """

    import torch as t
    from torchinfo import summary

    model_info = summary(
        model= model,
        input_size=input_size,
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    )
    return str(model_info)

