from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import os
import torch

from clothes.character import CharacterBuilder
from clothes.shirt_arm import Shirts
from clothes.shoes import Shoes
from clothes.pants import Pants
from clothes.hair import Hair
from clothes.face import CompleteFace
from tqdm import tqdm
import shutil 

def create_and_save_character(item_class, base_path, data_folder, item_name, index, resolution):
    
    item_folder = os.path.join(data_folder, item_name)
    os.makedirs(item_folder, exist_ok=True)
    item = item_class(base_path)

    if item_name == 'character':
        item.build_character()
        item.save_character(resolution=resolution, filename=os.path.join(data_folder, item_name, f"{item_name}_{index}.png"))
    else:
        character_image = Image.new('RGBA', (item.width, item.height), (255, 255, 255, 0))
        item.add_to_character(character_image)
        resized_image = character_image.resize(resolution, Image.Resampling.LANCZOS)

        resized_image.save(os.path.join(item_folder, f"{item_name}_{index}.png"))


class DataLoaderDiffusion(Dataset):
    def __init__(self, data_folder, resolution=(100, 100)):
        self.data_folder = data_folder
        self.resolution = resolution
        self.character_ids = sorted([
            f.split("_")[-1].split(".")[0] 
            for f in os.listdir(os.path.join(data_folder, "face"))
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

        layers = ["character", "shirt", "shoe", "pants", "hair", "face"]
        layer_list = []
        
        for layer in layers:
            layer_path = os.path.join(self.data_folder, layer, f"{layer}_{char_id}.png")
            layer_img = self.transform(Image.open(layer_path).convert("RGBA"))
            layer_list.append(layer_img)

        layer_tensor = torch.stack(layer_list)

        return layer_tensor 

if __name__ == "__main__":
    num_items = 16
    base_path = "../PNG"
    data_folder = "../data_diffusion"

    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)
    
    os.makedirs(data_folder, exist_ok=True)

    for i in tqdm(range(num_items), desc='Generating layers in characters'):

        resolution = (100,100)
        items = [
            (CharacterBuilder, "character"),
            (Shirts, "shirt"),
            (Shoes, "shoe"),
            (Pants, "pants"),
            (Hair, "hair"),
            (CompleteFace, "face")
        ]

        for item_class, item_name in items:
            create_and_save_character(item_class, base_path, data_folder, item_name, i, resolution)

    data_folder = "../data_diffusion"
    dataset = DataLoaderDiffusion(data_folder)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for batch_idx, layer_tensor in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Layer Tensor Shape: {layer_tensor.shape}") 
        
        first_layer_tensor = layer_tensor[0] 

        num_layers = first_layer_tensor.shape[0]
        fig, axes = plt.subplots(1, num_layers, figsize=(15, 5))
        for i in range(num_layers):
            layer_img = first_layer_tensor[i]  
            axes[i].imshow(F.to_pil_image(layer_img)) 
            axes[i].set_title(f"Layer {i + 1}")
            axes[i].axis("off")
        plt.show()

        break