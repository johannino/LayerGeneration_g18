import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel
from CharacterLoader import CharacterLayerLoader
from Discriminator import LayerDiscriminator
from Loss_functions import color_histogram_loss
from tqdm import tqdm

# --- 1. Create your dataset and dataloader ---
dataset = CharacterLayerLoader(data_folder="../data")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=3, pin_memory=True)

# --- 2. Initialize the UNet2DConditionModel ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

cross_attention_dim = 128 

generator = UNet2DConditionModel(
    sample_size=100,          # height and width of images (after resize)
    in_channels=3,            # RGB input
    out_channels=3,           # RGB output
    layers_per_block=2,
    block_out_channels=(64, 128, 128, 256),  
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    cross_attention_dim=cross_attention_dim,
).to(device)

discriminator = LayerDiscriminator(
        image_size=100,
        in_channels=3
    ).to(device)

# --- 3. Set optimizer and loss ---
criterion_mse = nn.MSELoss()
criterion_l1 = nn.L1Loss()
criterion_bce = nn.BCELoss()
criterion_layer = nn.CrossEntropyLoss()

lambda_adv = 1.0      # Adversarial loss weight
lambda_rec = 10.0     # Reconstruction loss weight
lambda_fm = 10.0      # Feature matching loss weight
lambda_layer = 2.0    # Layer classification loss weight

optimizer_G= optim.Adam(generator.parameters(), lr=1e-4)
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)
scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=200)
residual_reg_weight = 0.1  # Weight for residual regularization.
adversarial_weight = 0.5

# --- 4. Training loop ---
num_epochs = 50
real_label, fake_label = 1.0, 0.0
generator.train()
discriminator.train()

for epoch in range(num_epochs):
    
    running_g_loss = 0.0
    running_d_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        all_layers, _ = batch  # (batch_size, 6, 3, 100, 100)
        all_layers = all_layers.to(device)
        batch_size, num_layers, channels, height, width = all_layers.shape

        real_target = torch.full((batch_size, 1, 3, 3), real_label, device=device)
        fake_target = torch.full((batch_size, 1, 3, 3), fake_label, device=device)
        layer_labels = [torch.full((batch_size,), i, dtype=torch.long, device=device) for i in range(5)]

        for i in range(num_layers - 1):
            current_layer = all_layers[:, i, :, :, :]     # (batch_size, 3, H, W)
            next_layer = all_layers[:, i+1, :, :, :]       # (batch_size, 3, H, W)

            # Train discriminator
            optimizer_D.zero_grad()

            # Train with real samples
            real_validity, real_layer_pred, real_features = discriminator(next_layer, current_layer)
            d_real_loss = criterion_bce(real_validity, real_target)
            d_layer_loss_real = criterion_layer(real_layer_pred.view(batch_size, -1), layer_labels[i])

            # Add noise and generate fake sample
            noise = torch.randn_like(current_layer) * 0.1
            noisy_input = current_layer + noise
            condition = torch.zeros((batch_size, 1, cross_attention_dim), device=device)

            fake_next_layer = generator(
                sample=noisy_input,
                timestep=torch.zeros(batch_size, device=device, dtype=torch.long),
                encoder_hidden_states=condition
            ).sample

            # Train with fake samples
            fake_validity, _, _ = discriminator(fake_next_layer.detach(), current_layer)
            d_fake_loss = criterion_bce(fake_validity, fake_target)

            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss + lambda_layer * d_layer_loss_real
            d_loss.backward()
            optimizer_D.step()

            # Train generator
            optimizer_G.zero_grad()

            # Generate fake samples again for generator training
            fake_validity, fake_layer_pred, fake_features = discriminator(fake_next_layer, current_layer)

            # Calculate generator losses
            g_adversarial = criterion_bce(fake_validity, real_target)
            g_layer_loss = criterion_layer(fake_layer_pred.view(batch_size, -1), layer_labels[i])
            g_rec_loss = criterion_mse(fake_next_layer, next_layer)
            fm_loss = criterion_l1(fake_features, real_features.detach())

            # Add L1 loss to enforce sparsity
            g_rec_loss += residual_reg_weight * criterion_l1(
                fake_next_layer - current_layer, 
                torch.zeros_like(fake_next_layer)
            )

            #g_rec_loss += color_histogram_loss(fake_next_layer, next_layer)

            # Combine all losses
            g_loss = (lambda_adv * g_adversarial + 
                     lambda_rec * g_rec_loss + 
                     lambda_fm * fm_loss +
                     lambda_layer * g_layer_loss
                     )

            g_loss.backward()
            optimizer_G.step()

            running_g_loss += g_loss.item()
            running_d_loss += d_loss.item()
        
    avg_g_loss = running_g_loss / (len(dataloader) * (num_layers - 1))
    avg_d_loss = running_d_loss / (len(dataloader) * (num_layers - 1))
    print(f"Epoch [{epoch+1}/{num_epochs}] - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")

    scheduler_G.step()


print("Training complete!")


import matplotlib.pyplot as plt

# --- Pick a random sample ---
import random
generator.eval()  # Important: eval mode
idx = random.randint(0, len(dataset) - 1)
layer_sequence, _ = dataset[idx]  # (6, 3, 100, 100)

layer_sequence = layer_sequence.unsqueeze(0).to(device) 
num_layers = layer_sequence.shape[1]

# --- Start with the first layer ---
current_layer = layer_sequence[:, 0, :, :, :]  # (1, 3, 100, 100)

generated_layers = [current_layer.squeeze(0).cpu()]  # Save initial

# --- Predict step-by-step ---
for i in range(1, num_layers):
    # Noise (optional, depends on your training)
    noise = torch.randn_like(current_layer) * 0.1
    noisy_input = current_layer + noise

    # show the noisy input
    noisy_input_np = noisy_input.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    plt.imshow(noisy_input_np)
    plt.savefig(f'noisy_input_{i}.png')
    plt.close()

    condition = torch.zeros((1, 1, cross_attention_dim), device=device)

    with torch.no_grad():

        predicted = generator(
            sample=noisy_input,
            timestep=torch.zeros(1, device=device, dtype=torch.long),
            encoder_hidden_states=condition
        ).sample

    generated_layers.append(predicted.squeeze(0).cpu())  # Save
    current_layer = predicted  # Next input is the last prediction

# --- Plot the layers ---
fig, axs = plt.subplots(1, num_layers, figsize=(15, 5))

for i, layer in enumerate(generated_layers):
    img = layer.permute(1, 2, 0).clamp(0, 1)  # (H, W, 3)
    axs[i].imshow(img)
    axs[i].axis('off')
    axs[i].set_title(f"Layer {i}")

plt.tight_layout()
plt.savefig('generated_layers.png')
