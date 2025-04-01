import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
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

if __name__ == "__main__":

    data_folder = "../../data"
    dataset = CharacterLayerLoader(data_folder)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch_idx, (layer_tensor, full_image) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Layer Tensor Shape: {layer_tensor.shape}") 
        print(f"Full Image Shape: {full_image.shape}")     
        
        first_layer_tensor = layer_tensor[0] 
        first_full_image = full_image[0]     

        plt.figure(figsize=(8, 8))
        plt.imshow(F.to_pil_image(first_full_image))  
        plt.title("Full Image")
        plt.axis("off")
        plt.show()

        num_layers = first_layer_tensor.shape[0]
        fig, axes = plt.subplots(1, num_layers, figsize=(15, 5))
        for i in range(num_layers):
            layer_img = first_layer_tensor[i]  
            axes[i].imshow(F.to_pil_image(layer_img)) 
            axes[i].set_title(f"Layer {i + 1}")
            axes[i].axis("off")
        plt.show()

        break