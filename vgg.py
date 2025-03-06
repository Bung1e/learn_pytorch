import torch
import torch.nn as nn
from torchvision import models

model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
''' нужно указать значение weights иначе будет использовать не обученая сеть '''
print(model)