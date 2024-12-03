import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn import LeakyReLU
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.vae import VAE
from PIL import Image

class ShapeImageDataset(Dataset):
    def __init__(self, df_path, image_dir, transform=None):

        self.df = pd.read_csv(df_path) 
        self.image_dir = image_dir  
        self.transform = transform  
        self.emotion_columns = self.df.columns[7:]  

    def __len__(self):
        return min(len(self.df), len(os.listdir(self.image_dir)))

    def __getitem__(self, idx):

        adjusted_idx = idx + 1

        if adjusted_idx > len(self.df) or adjusted_idx > len(os.listdir(self.image_dir)):
            raise IndexError("Index out of range for dataset.")


        row = self.df.iloc[adjusted_idx - 1]
        image_path = os.path.join(self.image_dir, f"{adjusted_idx}.png")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new("RGB", (200, 200), (255, 255, 255))

        if self.transform:
            image = self.transform(image)

        labels = row[self.emotion_columns].fillna(0).values.astype(np.float32)
        label_tensor = torch.tensor(labels, dtype=torch.float32)

        return image, label_tensor