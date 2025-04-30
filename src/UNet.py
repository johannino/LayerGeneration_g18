import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel, UNet2DModel
from CharacterLoader import CharacterLayerLoader
from Discriminator import LayerDiscriminator
from Loss_functions import color_histogram_loss, total_variation_loss, gradient_penalty
from tqdm import tqdm

# --- 1. Create your dataset and dataloader ---
dataset = CharacterLayerLoader(data_folder="../data")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=3, pin_memory=True)

# --- 2. Initialize the UNet2DConditionModel ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

generator = UNet2DModel(
    sample_size=256,          # height and width of images (after resize)
    in_channels=3,            # RGB input
    out_channels=3,           # RGB output
    layers_per_block=1,
    block_out_channels=(32, 64, 64, 128),  
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"), #  "AttnDownBlock2D",
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
).to(device)

discriminator = LayerDiscriminator(
        image_size=256,
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
lambda_tv = 1        # Total variation loss weight

optimizer_G= optim.Adam(generator.parameters(), lr=1e-4)
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)
scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=200)
residual_reg_weight = 0.1  # Weight for residual regularization.
adversarial_weight = 0.5

# --- 4. Training loop ---
num_epochs = 30
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

# ...existing code...

        for i in range(num_layers - 1):
            current_layer = all_layers[:, i, :, :, :]     # (batch_size, 3, H, W)
            next_layer = all_layers[:, i+1, :, :, :]      # (batch_size, 3, H, W)
            
            # Calculate the target residual
            target_residual = next_layer - current_layer

            # Train discriminator
            optimizer_D.zero_grad()

            # Train with real samples (using actual next layer)
            real_validity, real_layer_pred, real_features = discriminator(next_layer, current_layer)
            d_real_loss = criterion_bce(real_validity, real_target)
            d_layer_loss_real = criterion_layer(real_layer_pred.view(batch_size, -1), layer_labels[i])

            # Generate residual
            noise = torch.randn_like(current_layer) * 0.1
            noisy_input = current_layer + noise

            # Generate residual instead of full layer
            predicted_residual = generator(
                sample=noisy_input,
                timestep=torch.zeros(batch_size, device=device, dtype=torch.long),
            ).sample

            # Create fake next layer by adding residual to current layer
            fake_next_layer = current_layer + predicted_residual

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
            
            # Use MSE loss on residuals instead of full layers
            g_rec_loss = criterion_mse(predicted_residual, target_residual)
            fm_loss = criterion_l1(fake_features, real_features.detach())
            tv_loss = total_variation_loss(predicted_residual)

            # Add L1 loss to enforce sparsity directly on the residual
            g_rec_loss += residual_reg_weight * criterion_l1(predicted_residual, torch.zeros_like(predicted_residual))

            # Add color histogram loss on the final output
            g_rec_loss += color_histogram_loss(fake_next_layer, next_layer)

            # Combine all losses
            g_loss = (lambda_adv * g_adversarial + 
                    lambda_rec * g_rec_loss + 
                    lambda_fm * fm_loss +
                    lambda_layer * g_layer_loss + 
                    lambda_tv * tv_loss)

            g_loss.backward()
            optimizer_G.step()

            running_g_loss += g_loss.item()
            running_d_loss += d_loss.item()
        
    avg_g_loss = running_g_loss / (len(dataloader) * (num_layers - 1))
    avg_d_loss = running_d_loss / (len(dataloader) * (num_layers - 1))
    print(f"Epoch [{epoch+1}/{num_epochs}] - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")

    scheduler_G.step()

    # save best model
    if (epoch + 1) % 5 == 0:
        torch.save(generator.state_dict(), f"../models/unet_model.pth")

    # --- Validation plotting code ---
    import matplotlib.pyplot as plt
    import random

    # Set model to evaluation mode
    generator.eval()

    # Pick a random sample
    idx = random.randint(0, len(dataset) - 1)
    layer_sequence, _ = dataset[idx]  # (6, 3, 256, 256)
    layer_sequence = layer_sequence.unsqueeze(0).to(device) 
    num_layers = layer_sequence.shape[1]

    # Start with the first layer
    current_layer = layer_sequence[:, 0, :, :, :]  # (1, 3, 256, 256)
    generated_layers = [current_layer.squeeze(0).cpu()]  # Save initial layer

    # Predict residuals step-by-step
    for i in range(1, num_layers):
        # Add noise to input
        noise = torch.randn_like(current_layer) * 0.1
        noisy_input = current_layer + noise
        
        with torch.no_grad():
            # Predict residual
            predicted_residual = generator(
                sample=noisy_input,
                timestep=torch.zeros(1, device=device, dtype=torch.long),
            ).sample
            
            # Add residual to current layer to get next layer
            next_layer = current_layer + predicted_residual
            generated_layers.append(next_layer.squeeze(0).cpu())
            
            # Update current layer for next iteration
            current_layer = next_layer

    # Plot both original and generated sequences
    fig, (ax1, ax2) = plt.subplots(2, num_layers, figsize=(20, 8))

    # Plot original sequence
    for i in range(num_layers):
        original = layer_sequence[0, i].cpu().permute(1, 2, 0).clamp(0, 1)
        ax1[i].imshow(original)
        ax1[i].axis('off')
        ax1[i].set_title(f"Original {i}")

    # Plot generated sequence
    for i, layer in enumerate(generated_layers):
        generated = layer.permute(1, 2, 0).clamp(0, 1)
        ax2[i].imshow(generated)
        ax2[i].axis('off')
        ax2[i].set_title(f"Generated {i}")

    plt.suptitle("Original vs Generated Layer Sequence")
    plt.tight_layout()
    if epoch % 5 == 0:
        plt.savefig(f'validation_comparison_unet_{epoch}.png')
    plt.close()

    # Optional: Plot residuals
    fig, axs = plt.subplots(1, num_layers-1, figsize=(15, 4))
    for i in range(num_layers-1):
        residual = (generated_layers[i+1] - generated_layers[i]).permute(1, 2, 0)
        # Scale residuals to visible range
        residual = (residual - residual.min()) / (residual.max() - residual.min())
        axs[i].imshow(residual)
        axs[i].axis('off')
        axs[i].set_title(f"Residual {i}")

    plt.suptitle("Generated Residuals Between Layers")
    plt.tight_layout()
    plt.savefig('residuals_unet.png')
    plt.close()

    # Return to training mode
    generator.train()