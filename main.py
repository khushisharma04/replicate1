#Load libraries
import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import variable
import torchvision 
import pathlib
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from tqdm import tqdm

class ShapeDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir):
        self.annotation = torch.zeros(14970)
        self.annotation[:3720] = 0
        self.annotation[3720:7485] = 1
        self.annotation[7485:11250] = 2
        self.annotation[11250:14970] = 3
        self.annotation = F.one_hot(self.annotation.to(torch.int64), 4)
        self.img_dir = img_dir
        self.transform = transforms.Compose ([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize ([0.5], [0.5]) #0-1 to [-1,1], formula (x-mean)/std
        ])
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        label = self.annotation[idx]
        if label.eq(torch.tensor([1, 0, 0, 0], dtype=torch.int64)).all():
            img_path = os.path.join(self.img_dir, 'circle', f"{idx}.png")
        elif label.eq(torch.tensor([0, 1, 0, 0], dtype=torch.int64)).all():
            img_path = os.path.join(self.img_dir, 'square', f"{idx - 3720}.png")
        elif label.eq(torch.tensor([0, 0, 1, 0], dtype=torch.int64)).all():
            img_path = os.path.join(self.img_dir, 'star', f"{idx - 7485}.png")
        elif label.eq(torch.tensor([0, 0, 0, 1], dtype=torch.int64)).all():
            img_path = os.path.join(self.img_dir, 'triangle', f"{idx - 11250}.png")
        image = torchvision.io.read_image(img_path)
        image = self.transform(image)
        return image, label

dataset = ShapeDataset('/Users/khushisharma/Code/replicate1/data/archive/shapes/')
train_dataset, valid_dataset = train_test_split(dataset, train_size=0.7, test_size=0.3)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)

#CNN Network


class ConvNet(nn.Module):
    def __init__(self,num_classes=4):
        super().__init__()
            
            #output size after convolution filter
            #((w-f+2P)/s) +1

            #Input shape= (64,3,200,200)

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=12, kernel_size=3,stride=1,padding=1)
            #shape= (64,12,200,200)
        self.bn1 = nn.BatchNorm2d(num_features=12)
            #shape= (64,12,200,200)
        self.relu1 = nn.ReLU()
            #shape= (64,12,200,200)

        self.pool=nn.MaxPool2d(kernel_size=2)
            #Reduce the image size by a factor of 2
            #shape= (64,12,100,100)

        self.conv2=nn.Conv2d(in_channels=12,out_channels=20, kernel_size=3,stride=1,padding=1)
            #shape= (64,20,100,100)
        self.relu2 = nn.ReLU()
            #shape= (64,20,100,100)

        self.conv3=nn.Conv2d(in_channels=20,out_channels=32, kernel_size=3,stride=1,padding=1)
            #shape= (64,32,100,100)
        self.bn3=nn.BatchNorm2d(num_features=32)
            #shape= (64,32,100,100)
        self.relu3 = nn.ReLU()
            #shape= (64,32,100,100)

        self.fc = nn.Linear(in_features=32*100*100, out_features=4)

            #feed forward function

    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)

        output=self.pool(output)
        
        output=self.conv2(output)
        output=self.relu2(output)

        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)

        #above output will be in matrix form, with shape of (64,32,100,100)

        output=output.view(-1,32*100*100)

        output=self.fc(output)

        return output

model = ConvNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for x, y in tqdm(train_loader):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y.to(torch.float32))
    loss.backward()
    optimizer.step()

model.eval()
avg_loss = 0
for x, y in valid_loader:
    output = model(x)
    loss = criterion(output.squeeze(1), y.to(torch.float32))
    avg_loss += loss.item()
avg_loss /= len(valid_loader)

print("VALIDATION LOSS IS " + avg_loss)