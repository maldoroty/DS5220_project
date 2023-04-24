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
    with open(data_name, 'w') as data_storage:
        data_storage.write('epoch,loss\n')
        for epoch in range(500):
            last_loss = np.inf
            total_loss = 0.0
            for i, batch in enumerate(train_data_loader):
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                out = model(images)
                loss = loss_function(out, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if i % checkpoint_size == checkpoint_size - 1:
                    print(f'\r[{epoch + 1}, {i + 1:5d}] loss: {total_loss / checkpoint_size:.3f}', end = "")
                    if total_loss < last_loss:
                        with open(f'model_weights_{save_name}_checkpoint.pth', 'wb') as weights:
                            torch.save(model.state_dict(), weights)
                    last_loss = total_loss
                    total_loss = 0.0
            data_storage.write(f'{epoch + 1},{total_loss / checkpoint_size:.3f}\n')

        print('\nFinished Training')

    torch.save(model.state_dict(), f'model_weights_{save_name}_final.pth')

BATCH_SIZE=512
if __name__ == "__main__":

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = CIFAR10(root=ROOT_PATH, download=True, train=True, transform=transform)
    eval_dataset = CIFAR10(root=ROOT_PATH, train=False, transform=transform)

    train_data_loader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    eval_data_loader = DataLoader(dataset=eval_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)

    model = resnet18(weights = None, num_classes = 10)
    
    if exists('model_weights_cnn_final.pth'):
        model.load_state_dict(torch.load('model_weights_cnn_final.pth'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    model.train()
    model.to(device)
    CHECKPOINT_SIZE = int(len(train_data_loader) / 5)

    train(train_data_loader, model, device, loss_function, optimizer, CHECKPOINT_SIZE, "cnn", 'cnn_data.csv')

    torch.save(optimizer.state_dict(), "cnn_optimizer_state.pth")
