# dependencies
import torch
import numpy as np
from torchvision.models import resnet18
import torch.optim as optim
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from os.path import dirname, exists

ROOT_PATH = dirname(__file__)



def train(train_data_loader, model, device, loss_function, optimizer, checkpoint_size, save_name, data_name):
    """
    Train a PyTorch model using a given data loader, loss function, and optimizer.

    Args:
        train_data_loader (torch.utils.data.DataLoader): A PyTorch DataLoader object containing the training data.
        model (torch.nn.Module): The PyTorch model to train.
        device (torch.device): The device to use for training (e.g. "cuda:0" for GPU or "cpu" for CPU).
        loss_function (torch.nn.Module): The loss function to use for training.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        checkpoint_size (int): The number of batches to train on before printing a checkpoint message.
        save_name (str): The base filename to use when saving checkpoint and final model weights.
        data_name (str): The filename to use when saving training loss data.

    Returns:
        None
    """
    # open file to store training data
    with open(data_name, 'w') as data_storage:
        data_storage.write('epoch,loss\n')
    # train over epochs
    for epoch in range(100):
        # initialize loss values
        last_loss = np.inf
        total_loss = 0.0
        # save model weights after each training epoch
        with open(f'model_weights_{save_name}_checkpoint.pth', 'wb') as weights:
            torch.save(model.state_dict(), weights)
        # iterate over all batches
        for i, batch in enumerate(train_data_loader):
            # split images and labels move to GPU
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            # zero gradients
            optimizer.zero_grad()
            # forward propagation
            out = model(images)
            loss = loss_function(out, labels)
            # backward propagation
            loss.backward()
            # update weights
            optimizer.step()
            total_loss += loss.item()
            # print training progress
            if i % checkpoint_size == checkpoint_size - 1:
                print(f'\r[{epoch + 1}, {i + 1:5d}] loss: {total_loss / checkpoint_size:.3f}', end = "")
        # save training data
        with open(data_name, "a") as data_storage:
            data_storage.write(f'{epoch + 1},{total_loss / checkpoint_size:.3f}\n')

    print('\nFinished Training')
    # save final model weights
    torch.save(model.state_dict(), f'model_weights_{save_name}_final.pth')

BATCH_SIZE=256
if __name__ == "__main__":

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = CIFAR10(root=ROOT_PATH, download=True, train=True, transform=transform)
    eval_dataset = CIFAR10(root=ROOT_PATH, train=False, transform=transform)

    train_data_loader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    eval_data_loader = DataLoader(dataset=eval_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
    
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(128*4*4, 512),
        nn.ReLU(),
        nn.Linear(512, 10))
    
    
    if exists('model_weights_cnn_final.pth'):
        model.load_state_dict(torch.load('model_weights_cnn_final.pth'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    model.train()
    model.to(device)
    CHECKPOINT_SIZE = int(len(train_data_loader) / 5)

    train(train_data_loader, model, device, loss_function, optimizer, CHECKPOINT_SIZE, "cnn", 'cnn_data.csv')

    torch.save(optimizer.state_dict(), "cnn_optimizer_state.pth")
