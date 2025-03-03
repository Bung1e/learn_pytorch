import os 
import json
from PIL import Image
import matplotlib.pyplot as plt

import torchvision
import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torchvision.datasets import ImageFolder

# можно делать свои собственные преобразования если не хватает уже готовых
# вот пример 
# class RavelTransform(nn.Model):
#     def forward(self, item):
#         return item.ravel()

# class DigitDataset(data.Dataset):
#     def __init__(self, path, train=True, transform=None):
#         self.path = os.path.join(path, "train" if train else "test")
#         self.transform = transform

#         with open(os.path.join(path, "format.json"), "r") as fp:
#             self.format = json.load(fp)

#         self.length = 0
#         self.files = []
#         self.targets = torch.eye(10)

#         for _dir, _target in self.format.items():
#                 path = os.path.join(self.path, _dir)
#                 list_files = os.listdir(path)
#                 self.length += len(list_files)
#                 self.files.extend(map(lambda _x: (os.path.join(path, _x), _target), list_files))

#     def __getitem__(self, item):
#         path_file, target = self.files[item]
#         t = self.targets[target]
#         img = Image.open(path_file)

#         if self.transform:
#             img = self.transform(img).ravel().float() / 255.0  # change to 1D tensor

#         return img, t

#     def __len__(self):
#         return self.length


class DigitNN(nn.Module):
    def __init__(self, input_dim, num_hidden, output_dim): 
        super().__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden)
        self.layer2 = nn.Linear(num_hidden, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        return x

model = DigitNN(28 * 28, 32, 10)

state_dict = torch.load(r'edu/PyTorch/model_dnn.tar', weights_only=True) # loading states of nn
model.load_state_dict(state_dict) # loading states to model

transforms = tfs.Compose([tfs.ToImage(), tfs.Grayscale(), tfs.ToDtype(torch.float32, scale=True), tfs.Lambda(lambda _img: _img.ravel())]) 
# tfs.Compose - позволяет применять несколько преобразований
# мы добавили tfs.Grayscale() потому что до этого размерность была [3, 28, 28] 3 означает что изображение в формате RGB
# а у нас серое изображение поэтому к такому размеру и надо привести, в итоге получаем [1, 28, 28]
# scale означает что мы нормируем изображение (приводим в шкалу от 0 до 1)

# d_train = ImageFolder("edu/PyTorch/dataset/train", transform=transforms)  
# используем ImageFolder вместо целого класса DigitDataset написаного ранее, функция ImageFolder делает тоже самое

# train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)   
# print(len(train_data))

dataset_mnist = torchvision.datasets.MNIST(r'edu\PyTorch\datasets\mnist', download=True, train=True, transform=transforms)
d_train, d_val = data.random_split(dataset_mnist, [0.7, 0.3])
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)
train_data_val = data.DataLoader(d_val, batch_size=32, shuffle=False)
# когда проверяем перемешивать не надо


optimizer = optim.Adam(params=model.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()
epochs = 15

loss_lst_val = [] # список значений потерь при валидации
loss_lst = [] # список значений потерь при обучении

for _e in range(epochs):
    model.train()
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=False)
    for x_train, y_train in train_data:
        predict = model(x_train)
        loss = loss_function(predict, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1/lm_count * loss.item() + (1 - 1/lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean={loss_mean:.3f}")

    model.eval()
    Q_val = 0
    count_val = 0

    for x_val, y_val in train_data_val:
        with torch.no_grad():
            p = model(x_val)
            loss = loss_function(p, y_val)
            Q_val += loss.item()
            count_val += 1

    Q_val /= count_val

    loss_lst.append(loss_mean)
    loss_lst_val.append(Q_val)

    print(f" | loss_mean={loss_mean:.3f}, Q_val={Q_val:.3f}")



d_test = ImageFolder(r'edu/PyTorch/dataset/test', transform=transforms)  
test_data = data.DataLoader(d_test, batch_size=500, shuffle=False)  #перемешивание отключено потому что для тестирования это не имеет значения
Q = 0
model.eval()

for x_test, y_test in test_data:
    with torch.no_grad(): #градиет нужен только для обучения
        p = model(x_test) #проходим по модели
        p = torch.argmax(p, dim=1) 
        # torch.argmax(p, dim=1) выбирает индекс класса с наибольшей вероятностью (по столбцам, если dim=1). 
        # Теперь p — вектор с индексами предсказанных классов. 
        Q += torch.sum(p == y_test).item()

Q /= len(d_test)
print(Q)

# st = model.state_dict()
# torch.save(st, 'edu/pytorch/model_dnn.tar')
# state_dict = torch.load('model_dnn.tar', weights_only=True) loading states of nn
# model.load_state_dict(state_dict) loading states to model

plt.plot(loss_lst)
plt.plot(loss_lst_val)
plt.grid()
plt.show()