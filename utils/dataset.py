import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class ShapeImageDataset(Dataset):
    def __init__(self, df_path, image_dir, transform=None):
        self.df = pd.read_csv(df_path)  
        self.image_dir = image_dir  
        self.transform = transform  

    def __len__(self):
        return len(self.df)  

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        
    
        image_filename = f"{row['Index']}.png" 
        image_path = os.path.join(self.image_dir, image_filename)

        try:
   
            image = Image.open(image_path).convert("RGB")
        except IOError as e:
            print(f"Error loading image {image_path}: {e}")

            image = Image.new("RGB", (200, 200), (255, 255, 255))  


        if self.transform:
            image = self.transform(image)

        return image
