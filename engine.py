
"""
Contains the functions for training, validating, and testng a model. Requires importing torch and summary from torchinfo beforehand. 
"""

import torch
from torch import nn
from typing import Dict, List, Tuple
from timeit import default_timer as timer
import os
import json
from pathlib import Path


from sys import modules
module_name = 'torchinfo'


try:
    from torchinfo import summary
except ModuleNotFoundError:
    print(f"Haven't imported torchinfo!")
    print(f"Without torchinfo, module will misbehave later on!")


def accuracy_function(preds, true_label):
    """A function to calculate the accuracy of a batch of predictions.
    Args:
    preds: A tensor of predictions.
    true_label: A tensor of true labels.
    Returns:
    A float representing the accuracy (%)
    """
    correct = torch.eq(preds, true_label).sum().item()
    acc = (correct/len(preds))*100
    return acc


def training_step(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim,
        device: torch.device,
        accuracy_fn = accuracy_function,
        random_seed = None,
        testing_mode = False,
):
    """
    Trains a Pytorch model for one epoch.

    Args:
    model: A Pytorch model to train.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A Pytorch loss function.
    optimizer: A Pytorch optimizer.
    accuracy_fn: A function that calculates the accuracy. (Should output %)
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    random_seed: Determine the random seed of the training step. Only affect Ptorch stuff. 
    testing_mode: whether to use testing mode. print out data every batch. 
    Returns:
    A dictionary of training loss, training accuracy, the time taken to train, the number of trained epochs, and the model summary.
    """
    
    if random_seed != None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
    train_loss, train_acc = 0,0
    # image, label = next(iter(dataloader))
    # summary = str(summary(model, input_size = image.shape, device = device))

    results = {
        'train_loss': 0,
        'train_acc': 0,
    }

    for batch, (image, label) in enumerate(dataloader):
        image, label = image.to(device), label.to(device)

        model.train()

        logits = model(image)

        preds = torch.argmax(torch.softmax(logits, dim = 1), dim = 1)

        loss = loss_fn(logits, label)

        train_loss += loss.item() * image.size(0) #calculate total loss, accoutning for batch size

        train_acc += accuracy_fn(preds, label) * image.size(0)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        if testing_mode == True:
            print(f"Batch: {batch} | Train loss: {loss} | Train acc: {accuracy_fn(preds, label)}")

    train_loss /= len(dataloader.dataset)
    train_acc /= len(dataloader.dataset)

    results['train_loss'] = train_loss
    results['train_acc'] = train_acc

    return results


def validation_step(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim,
        device: torch.device,
        accuracy_fn = accuracy_function,
        random_seed = None,
):
    """
    Trains a Pytorch model for a specified number of epochs.

    Args:
    model: A Pytorch model to train.
    dataloader: A DataLoader instance for the model to be validated on.
    loss_fn: A Pytorch loss function.
    optimizer: A Pytorch optimizer.
    accuracy_fn: A function that calculates the accuracy. (Should output %)
    device: A target device to compute on (e.g. "cuda" or "cpu").
    random_seed: Determine the random seed of the validation step. Shouldn't be useful, but who knows. 
    Returns:
    A dictionary of validation loss, validation accuracy, and the time taken to validate
    """

    validation_loss, validation_acc = 0,0

    if random_seed != None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
    results = {
        'validation_loss': 0,
        'validation_acc': 0,
    }

    for batch, (image, label) in enumerate(dataloader):
        image, label = image.to(device), label.to(device)

        model.eval()

        with torch.inference_mode():

            logits = model(image)

            preds = torch.argmax(torch.softmax(logits, dim = 1), dim = 1)

            loss = loss_fn(logits, label)

            validation_loss += loss.item()*image.size(0)

            validation_acc += accuracy_fn(preds, label)*image.size(0)

    validation_loss /= len(dataloader.dataset)
    validation_acc /= len(dataloader.dataset)

    results['validation_loss'] = validation_loss
    results['validation_acc'] = validation_acc

    return results

class EarlyStopping():
    def __init__(self, min_delta: int, patience):
        """A class to implement stopping a model early.
        Args:
        min_delta: minimum change in loss to qualify as an improvement. (positive int value)
        patience: number of epochs to wait before stopping.
        """
        assert patience > 0, "Patience should be greater than 0"
        assert min_delta > 0, "Min delta should be greater than 0"
        self.best_model_params = None
        self.best_loss = None
        self.min_delta = min_delta
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss, model):
        if self.best_loss == None:
            self.best_loss = loss
            self.best_model_params = model.state_dict()
        elif self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            self.best_model_params = model.state_dict()
            self.counter = 0
        else:
            self.counter+= 1

            if self.counter >= self.patience:
                self.early_stop = True

            print(f"{self.counter} epoch(s) without improvement~~")
    def load_model_params(self, model):
        model.load_state_dict = self.best_model_params
        print("Reloaded best model parameters~")



def save_results_json(file_path: str, results):
    """
    Save a file oas a json file. 

    Args:
    file_name: the name of the file. 
    results: the content we want to save. Should be str. 
    """
    
    with open(file_path, 'w') as f:
        f.write(json.dumps(results))


def fitting_model(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    validation_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device,
    early_stop = True,
    testing_mode: bool = False,
    writer = None,
    lr_scheduler: torch.optim.lr_scheduler = None,
    random_seed = None,

):
    """Train and validate a model for an number of epochs.
    Passes a model through train step and validation step.

    Args:
    model: A Pytorch model.
    train_dataloader: A DataLoader instance for the model to be trained on.
    validation_dataloader: A DataLoader instance for the model to be validated on.
    optimizer: A Pytorch optimizer.
    loss_fn: A Pytorch loss function.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    early_stop: Whether to implement early stop. (Default is 4 epochs without imporvement before stopping. )
    testing_mode: Put function into testing mode, print train loss and accuracy every batch. 
    writer: whether to save the results of the experiments as tensorboard file. (torch.utils.tensorboard.SummaryWriter)
    lr_scheduler: learning rate scheduler. Use create partial class in utils, else gonna error. 
    random_seed: the random seed of the operation. defaults to None
    """

    start = timer()

    if lr_scheduler != None:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(train_dataloader), epochs=epochs)

    
    def _end_fitting_model(start, results):
        """
        A subfunction of the fitting_model function. used to end the fitting model_function

        Args:
        start: the start time of the fitting_model func, measured in seconds.
        results: the results dict of the fitting_model func

        Returns:
        results
        """

        if writer != None:

            log_dir = writer.log_dir
            print(str(log_dir))
            summary_path = os.path.join(log_dir, 'summary.json')

            save_results_json(file_path = summary_path, results = architecture)

        end = timer()

        time = end - start
        results['time'] = time
        print(f"Time elapsed: {time}") #time taken model to train
        
        return results
    
    image, label = next(iter(validation_dataloader))

    architecture = str(summary(model, input_size = image.shape, device = device))

    if early_stop == True:
        early_stopping = EarlyStopping(patience = 4, min_delta = 0.001)

    model.to(device)#put model in correct device

    results = {
        'summary': architecture,
        'train_loss': [],
        'train_acc': [],
        'validation_loss': [],
        'validation_acc': [],
        'epochs': 0,
        'device': device,
        'time': 0

    }
    for epoch in range(1, epochs + 1):
        train_results = training_step(
            model = model,
            dataloader = train_dataloader,
            loss_fn = loss_fn,
            optimizer = optimizer,
            device = device,
            random_seed = random_seed, 
            testing_mode = testing_mode
        )
        if lr_scheduler != None:
            lr_scheduler.step()
        
        params = optimizer.param_groups[0]

        print(f"After scheduler: {params['lr']}")

        print(f"\nEpoch: {epoch} | Train loss: {train_results['train_loss']} | Train accuracy: {train_results['train_acc']}")
        results['train_loss'].append(train_results['train_loss'])
        results['train_acc'].append(train_results['train_acc'])

        validation_results = validation_step(
            model = model,
            dataloader = validation_dataloader,
            loss_fn = loss_fn,
            optimizer = optimizer,
            device = device,
            random_seed = random_seed
        )

        print(f"Validation loss: {validation_results['validation_loss']} | Validation accuracy: {validation_results['validation_acc']}")
        results['validation_loss'].append(validation_results['validation_loss'])
        results['validation_acc'].append(validation_results['validation_acc'])
        results['epochs'] += 1

        if writer != None:
            writer.add_scalars(main_tag = "Accuracy",
                                tag_scalar_dict={
                                    'train_acc': train_results['train_acc'],
                                    'validation_acc': validation_results['validation_acc'],
                                },
                                global_step=epoch
            )

            writer.add_scalars(main_tag = "Loss",
                                tag_scalar_dict={
                                    'train_loss': train_results['train_loss'],
                                    'validation_loss': validation_results['validation_loss'],
                                },
                                global_step=epoch
            )
        

        if early_stop == True:
            early_stopping(validation_results['validation_loss'], model)
            if early_stopping.early_stop:
                print("Early stopping...")
                early_stopping.load_model_params(model)

                return _end_fitting_model(start = start, results = results)
     
    return _end_fitting_model(start = start, results = results)