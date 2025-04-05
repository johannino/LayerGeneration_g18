import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from CharacterLoader import CharacterLayerLoader

class LayerWiseARModel(nn.Module):
    def __init__(self, num_layers=5, hidden_dim=256):
        super().__init__()
        self.num_layers = num_layers


if __name__ == "__main__":

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    data_folder = "../../data"
    dataset = CharacterLayerLoader(data_folder)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LayerWiseARModel().to(device)
    exit()
    #train(model, dataloader)

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
        #plt.show()

        num_layers = first_layer_tensor.shape[0]
        fig, axes = plt.subplots(1, num_layers, figsize=(15, 5))
        for i in range(num_layers):
            layer_img = first_layer_tensor[i]  
            axes[i].imshow(F.to_pil_image(layer_img)) 
            axes[i].set_title(f"Layer {i + 1}")
            axes[i].axis("off")
        #plt.show()

        break

