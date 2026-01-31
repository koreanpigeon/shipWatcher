import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from src.data_preprocessing import train_samples, val_samples
from src.model_def import get_shipWatcher
from src.utils import device, class_list, pic_transform


# Create custom PyTorch dataset implementation utilising "lazy loading" for memory-efficient data pipelining

class Data(Dataset):
    def __init__(self, dataset, transform=None):
        self.x = dataset[0]
        self.y = dataset[1]
        self.transform = transform
        self.len = len(dataset[0])

    def __getitem__(self, index):
        x = self.transform(Image.open(self.x[index]).convert("RGBA").convert("RGB"))
        y = self.y[index]
        return x, y

    def __len__(self):
        return self.len


# Define function for training, validating shipWatcher model with certain hyperparameters

def prepare_and_train(epochs, lr, weight_decay, betas):
    shipWatcher = get_shipWatcher()
    shipWatcher = shipWatcher.to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(shipWatcher.fc.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    train_dataset = Data(train_samples(), transform=pic_transform)
    val_dataset = Data(val_samples(), transform=pic_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=5, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=5)

    for epoch in range(epochs):
        shipWatcher.train()
        train_loss = 0
        val_accuracy = 0
        print("_________________")
        print(f"Epoch {epoch+1}")
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimiser.zero_grad()
            yhat = shipWatcher(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
        print(f"Training Loss: {train_loss}")
        shipWatcher.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                yhat = shipWatcher(x)
                val_accuracy += (torch.argmax(yhat, dim=1)==y).sum().item()
        print(f"{val_accuracy}/{len(val_images)}")
        if epoch+1 == epochs:
            return train_loss, val_accuracy, shipWatcher.state_dict()
        else:
            pass
