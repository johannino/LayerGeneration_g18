import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Model.CharacterLoader import CharacterLayerLoader
from unet import ConditionalUNet
import matplotlib.pyplot as plt
import os

# Config
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
data_folder = "../data"
batch_size = 8
image_resolution = (100, 100)
num_epochs = 500
residual_weight = 0.1
teacher_forcing_until = 50

# Data
dataset = CharacterLayerLoader(data_folder, resolution=image_resolution)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
model = ConditionalUNet(
    in_channels=3,
    base_channels=128,
    time_embed_dim=256
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

# Training
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for layers, _ in dataloader:
        layers = layers.to(device)
        B, T, C, H, W = layers.shape

        optimizer.zero_grad()
        loss = 0

        pred_layers = [layers[:, 0]]

        for t in range(1, T):
            input_layer = layers[:, t-1] if epoch < teacher_forcing_until else pred_layers[-1].detach()
            target_layer = layers[:, t]
            timestep = torch.full((B,), t, dtype=torch.long, device=device)

            residual = model(input_layer, timestep) * 0.1
            output = model(input_layer, timestep)  # no double addition

            loss_main = criterion(output, target_layer)
            loss_res = residual.abs().mean()
            loss += loss_main + residual_weight * loss_res

            pred_layers.append(output)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")
    scheduler.step()

# Eval
model.eval()
with torch.no_grad():
    for layers, _ in dataloader:
        layers = layers.to(device)
        B, T, C, H, W = layers.shape

        preds = [layers[:, 0]]
        for t in range(1, T):
            input_layer = preds[-1]
            timestep = torch.full((B,), t, device=device)
            residual = model(input_layer, timestep)
            pred = torch.clamp(input_layer + residual, 0.0, 1.0)
            preds.append(pred)

        to_np = lambda x: x[0].permute(1, 2, 0).cpu().numpy()
        fig, axes = plt.subplots(2, T, figsize=(3*T, 6))

        for t in range(T):
            axes[0, t].imshow(to_np(layers[:, t]))
            axes[0, t].set_title(f"GT Layer {t+1}")
            axes[1, t].imshow(to_np(preds[t]))
            axes[1, t].set_title(f"Pred Layer {t+1}")
            axes[0, t].axis("off")
            axes[1, t].axis("off")

        plt.tight_layout()
        os.makedirs("../figures", exist_ok=True)
        plt.savefig("../figures/final_conditional_unet.png")
        plt.show()
        break