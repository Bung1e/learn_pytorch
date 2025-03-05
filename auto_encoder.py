import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os 
import json
from PIL import Image
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torch.optim as optim
from tqdm import tqdm
import torchvision

class autoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28)
        )

    def forward(self, x):
        x = self.encoder()
        x = self.decoder()
        return x 

transforms = tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True)])
mnist_train = torchvision.datasets.MNIST(r'edu\PyTorch\datasets\mnist', download=True, train=True, transform=transforms)
train_data = data.DataLoader(mnist_train, batch_size=32, shuffle=True)

mnist_test = torchvision.datasets.MNIST(r'edu\PyTorch\datasets\mnist', download=True, train=False, transform=transforms)
test_data = data.DataLoader(mnist_test, batch_size=32, shuffle=False)

optimizer = optim.Adam(params=model.parameters(), lr=0.01)
loss_function = nn.MSELoss()
epochs = 15

for _ in range(epochs):
    model.train()
    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_data:
        predict = model.predict(x_train())
        loss = loss_function(predict, x_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    

model = autoEncoder()
print(len(test_data))