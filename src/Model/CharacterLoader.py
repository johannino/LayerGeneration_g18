import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class CharacterLayerLoader(Dataset):
    def __init__(self, data_folder, resolution=(100, 100)):
        self.data_folder = data_folder
        self.resolution = resolution
        self.character_ids = sorted([
            f.split("_")[-1].split(".")[0] 
            for f in os.listdir(os.path.join(data_folder, "base"))
            if f.endswith(".png")
        ])
        self.transform = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.ToTensor(),  
        ])

    def __len__(self):
        return len(self.character_ids)

    def __getitem__(self, idx):
        char_id = self.character_ids[idx]
        
        base_path = os.path.join(self.data_folder, "base", f"base_character_{char_id}.png")
        full_image = self.transform(Image.open(base_path).convert("RGB"))
        
        layers = []
        layer_names = ["shirt", "shoe", "pants", "hair", "face"]
        
        for layer in layer_names:
            layer_path = os.path.join(self.data_folder, layer, f"{layer}_character_{char_id}.png")
            layer_img = self.transform(Image.open(layer_path).convert("RGB"))
            layers.append(layer_img)

        layer_tensor = torch.stack(layers, dim=0)

        return layer_tensor, full_image 
