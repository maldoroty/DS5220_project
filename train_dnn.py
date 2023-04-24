import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from os.path import dirname, exists
from train_cnn import train


ROOT_PATH = dirname(__file__)
BATCH_SIZE=512

if __name__ == "__main__":

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = CIFAR10(root=ROOT_PATH, download=True, train=True, transform=transform)
    eval_dataset = CIFAR10(root=ROOT_PATH, train=False, transform=transform)

    train_data_loader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    eval_data_loader = DataLoader(dataset=eval_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
    
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3072, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        nn.Sigmoid())
    
    if exists('model_weights_dnn_final.pth'):
        model.load_state_dict(torch.load('model_weights_dnn_final.pth'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    model.train()
    model.to(device)
    CHECKPOINT_SIZE = int(len(train_data_loader) / 5)

    train(train_data_loader, model, device, loss_function, optimizer, CHECKPOINT_SIZE, "dnn", "dnn_data.csv")

    torch.save(optimizer.state_dict(), "dnn_optimizer_state.pth")