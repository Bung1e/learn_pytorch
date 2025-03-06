import os 
import json 
from PIL import Image

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.v2 as tfs

model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2), 
    nn.Conv2d(32, 8, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2), 
    nn.Conv2d(8, 4, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(), 
    nn.Linear(4096, 128),
    nn.ReLU(),
    nn.Linear(128, 2)    
)

path = 'datasets/dataset_reg/test/'
num_img = 506

st = torch.load('model_sun.tar', weights_only=True)
model.load_state_dict(st)

with open(os.path.join(path, 'format.json'), 'r') as fp:
    format = json.load(fp)

transforms = tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True)])
img = Image.open(os.path.join(path,f'sun_reg_{num_img}.png')).convert('RGB')
img_t = transforms(img).unsqueeze(0)

model.eval()
predict = model(img_t)
p = predict.detach().squeeze().numpy() 

''' 
squeeze() удаляет все размеры тензора, которые равны 1. Например, если тензор имеет форму (1, C, H, W), 
то после применения squeeze() он станет иметь форму (C, H, W).
'''

plt.imshow(img)
plt.scatter(p[0], p[1], s=20, c='r')
plt.show()


