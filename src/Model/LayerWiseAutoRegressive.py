import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CharacterLoader import CharacterLayerLoader

class LayerWiseARModel(nn.Module):
    def __init__(self, num_layers=5, hidden_dim=256):
        super().__init__()
        self.num_layers = num_layers 
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim)
        self.decoder = nn.TransformerDecoder(self.decoder_layer)
        self.fc_out = ...
    
    def forward(self, x):
        x = self.decoder_layer(...)
        return x


if __name__ == "__main__":

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    data_folder = "../../data"
    dataset = CharacterLayerLoader(data_folder)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # Size of Batch: [32, 5, 3, 100, 100] -> [batch_size, num_layers, rgb, resolution, resolution]

    model = LayerWiseARModel().to(device)
    #train(model, dataloader)
