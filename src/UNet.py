import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from diffusers import UNet2DModel
from CharacterLoader import CharacterLayerLoader
from MulanDataset import MulanLayerDataset
from Discriminator import LayerDiscriminator
from Loss_functions import color_histogram_loss, total_variation_loss, gradient_penalty
from GAN_train import train_gan

# --- 1. Create your dataset and dataloader ---
dataset = CharacterLayerLoader(data_folder="../data")
#dataset = MulanLayerDataset('../MULAN_data')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=3, pin_memory=True)

# --- 2. Initialize the UNet2DConditionModel ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

generator = UNet2DModel(
    sample_size=256,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels= (64, 128, 256, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    mid_block_type="AttnMidBlock2D",
).to(device)

discriminator = LayerDiscriminator(
        image_size=256,
        in_channels=3
).to(device)


# --- 4. Training loop ---
if __name__ == '__main__':

    optimizer_G=optim.Adam(generator.parameters(), lr=1e-4)

    train_gan(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        device=device,
        num_epochs=6,
        lambda_adv=1.0,
        lambda_rec=10.0 ,
        lambda_fm=10.0  ,
        lambda_layer=2.0,
        lambda_tv=1 ,
        residual_reg_weight= 0.1 ,
        color_histogram_loss=color_histogram_loss,
        total_variation_loss=total_variation_loss,
        criterion_mse=nn.MSELoss(),
        criterion_l1=nn.L1Loss(),
        criterion_layer=nn.CrossEntropyLoss(),
        optimizer_G=optimizer_G,
        optimizer_D=optim.Adam(discriminator.parameters(), lr=1e-4),
        scheduler_G=optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=200),
        model_save_path="../models/unet_model.pth"
    )


