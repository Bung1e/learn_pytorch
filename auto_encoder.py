import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os 
import json
import matplotlib.pyplot as plt
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
        x = self.encoder(x)
        x = self.decoder(x)
        return x 

transforms = tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True), tfs.Lambda(lambda _img: _img.ravel())])
mnist_train = torchvision.datasets.MNIST(r'edu\PyTorch\datasets\mnist', download=True, train=True, transform=transforms)
d_train, d_val = data.random_split(mnist_train, [0.7, 0.3])

train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)
train_data_val = data.DataLoader(d_val, batch_size=32, shuffle=False)

mnist_test = torchvision.datasets.MNIST(r'edu\PyTorch\datasets\mnist', download=True, train=False, transform=transforms)
test_data = data.DataLoader(mnist_test, batch_size=32, shuffle=False)

model = autoEncoder()

# state_dict = torch.load(r'edu/PyTorch/autoencoder.tar', weights_only=True)
# model.load_state_dict(state_dict)

optimizer = optim.Adam(params=model.parameters(), lr=0.001)
loss_function = nn.MSELoss()
epochs = 15

for _ in range(epochs):
    model.train()
    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_data:
        predict = model(x_train)
        loss = loss_function(predict, x_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    for x_val, y_val in train_data_val:
        with torch.no_grad():
            p = model(x_val)
            loss = loss_function(p, x_val)
mse = 0
model.eval()
x_test, y_test = next(iter(test_data))
x_new = x_test[:5]
with torch.no_grad():
    x_reconstructed = model(x_test)
    mse += F.mse_loss(x_reconstructed, x_test, reduction='sum').item()

mse /= len(test_data.dataset)
print(f"Среднеквадратичная ошибка (MSE): {mse}")

x_test = x_test.view(-1, 28, 28)
x_reconstructed = x_reconstructed.view(-1, 28, 28)

num_images = 10  

fig, axes = plt.subplots(2, num_images, figsize=(num_images, 2))

for i in range(num_images):
    axes[0, i].imshow(x_test[i].cpu().numpy(), cmap="gray")
    axes[0, i].axis("off")

    axes[1, i].imshow(x_reconstructed[i].cpu().numpy(), cmap="gray")
    axes[1, i].axis("off")

plt.show()

# st = model.state_dict()
# torch.save(st, 'edu/pytorch/autoencoder.tar')

