import os
import pandas as pd
import numpy as np
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, img_folder, csv_folder, transform=None):
        self.img_folder = img_folder
        self.csv_folder = csv_folder
        self.transform = transform
        self.file_list = [f for f in os.listdir(img_folder) if f.endswith('.png')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        prefix = img_name.split('.')[0]
        csv_path = os.path.join(self.csv_folder, prefix + ".csv")
        img_path = os.path.join(self.img_folder, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            print(f"Error loading image: {img_path}, Error: {str(e)}")
            img = None

        matrix = pd.read_csv(csv_path, skiprows=1, header=None).values.astype(np.float32)


        return img, torch.tensor(matrix)

class MulDataset(Dataset):
    def __init__(self, img_folder, csv_folder, transform=None):
        self.img_folder = img_folder
        self.csv_folder = csv_folder
        self.transform = transform
        self.file_list = [f for f in os.listdir(img_folder) if f.endswith('.png')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        prefix = img_name.split('.')[0]
        csv_path = os.path.join(self.csv_folder, prefix + ".csv")
        img_path = os.path.join(self.img_folder, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            print(f"Error loading image: {img_path}, Error: {str(e)}")
            img = None

        matrix = pd.read_csv(csv_path, skiprows=1, header=None).values.astype(np.float32)
        # 计算残基个数
        residue_count = np.count_nonzero(matrix[0, :]) + 1

        return img, torch.tensor(matrix), torch.tensor(residue_count, dtype=torch.float32)



transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.scale = (in_channels // 8) ** -0.5

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)

        attention = torch.bmm(query, key) * self.scale
        attention = torch.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        return out + x  # skip connection