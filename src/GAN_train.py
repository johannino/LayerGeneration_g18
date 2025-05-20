from Loss_functions import color_histogram_loss, total_variation_loss
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

def train_gan(
    generator,
    discriminator,
    dataloader,
    device,
    num_epochs=6,
    lambda_adv=1.0,
    lambda_rec=10.0,
    lambda_fm=10.0,
    lambda_layer=2.0,
    lambda_tv=1,
    residual_reg_weight=0.1,
    color_histogram_loss=color_histogram_loss,
    total_variation_loss=total_variation_loss,
    criterion_mse=nn.MSELoss(),
    criterion_l1=nn.L1Loss(),
    criterion_layer=nn.CrossEntropyLoss(),
    optimizer_G=None,
    optimizer_D=None,
    scheduler_G=None,
    model_save_path="../models/unet_model.pth"
):
    real_label, fake_label = 1.0, 0.0
    generator.train()
    discriminator.train()
    g_losses = [np.inf]
    d_losses = [np.inf]
    criterion_bce = nn.BCELoss()

    is_unet = True if 'unet' in model_save_path else False

    for epoch in range(num_epochs):
        running_g_loss = 0.0
        running_d_loss = 0.0

        for batch, _ in dataloader:
            all_layers = batch.to(device)
            batch_size, num_layers, channels, height, width = all_layers.shape

            layer_labels = [torch.full((batch_size,), i, dtype=torch.long, device=device) for i in range(5)]

            if is_unet:
                for i in range(num_layers - 1):
                    current_layer = all_layers[:, i, :, :, :]
                    next_layer = all_layers[:, i+1, :, :, :]
                    target_residual = next_layer - current_layer

                    # --- Train discriminator ---
                    optimizer_D.zero_grad()
                    real_validity, real_layer_pred, real_features = discriminator(next_layer, current_layer)
                    d_real = real_validity.mean()

                    predicted_residual = generator(
                            sample=current_layer,
                            timestep=torch.zeros(batch_size, device=device, dtype=torch.long),
                        ).sample
                    fake_next_layer = current_layer + predicted_residual

                    fake_validity, _, _ = discriminator(fake_next_layer.detach(), current_layer)
                    d_fake = fake_validity.mean()

                    d_loss = -d_real + d_fake
                    d_layer_loss = criterion_layer(real_layer_pred.view(batch_size, -1), layer_labels[i])
                    d_loss += lambda_layer * d_layer_loss

                    d_loss.backward()
                    optimizer_D.step()

                    for p in discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)

                    # --- Train generator ---
                    optimizer_G.zero_grad()
                    fake_validity, fake_layer_pred, fake_features = discriminator(fake_next_layer, current_layer)
                    g_adversarial = -fake_validity.mean()
                    g_layer_loss = criterion_layer(fake_layer_pred.view(batch_size, -1), layer_labels[i])
                    g_rec_loss = criterion_mse(predicted_residual, target_residual)
                    fm_loss = criterion_l1(fake_features, real_features.detach())
                    tv_loss = total_variation_loss(predicted_residual)

                    g_rec_loss += residual_reg_weight * criterion_l1(predicted_residual, torch.zeros_like(predicted_residual))
                    g_rec_loss += color_histogram_loss(fake_next_layer, next_layer)

                    g_loss = (
                        lambda_adv * g_adversarial +
                        lambda_rec * g_rec_loss +
                        lambda_fm * fm_loss +
                        lambda_layer * g_layer_loss +
                        lambda_tv * tv_loss
                    )

                    g_loss.backward()
                    optimizer_G.step()

                    running_g_loss += g_loss.item()
                    running_d_loss += d_loss.item()
                
            else:
                layer1 = all_layers[:, 0]   # Base layer.
                gt_layers = [all_layers[:, i] for i in range(1, 6)]  # Layers 2-6

                # --- Train discriminator ---
                optimizer_D.zero_grad()
                layer_labels = [torch.full((batch_size,), i, dtype=torch.long, device=device) for i in range(5)]

                pred_layers = generator(
                    layer1=layer1, 
                    gt_layer2=all_layers[:, 1],
                    gt_layer3=all_layers[:, 2],
                    gt_layer4=all_layers[:, 3], 
                    gt_layer5=all_layers[:, 4],
                    teacher_forcing=True
                )

                d_real_loss = 0
                d_layer_loss_real = 0
                d_fake_loss = 0

                # Train with real samples
                for i, (gt_layer, layer_label) in enumerate(zip(gt_layers, layer_labels)):
                    condition = layer1 if i == 0 else gt_layers[i-1]
                    
                    real_validity, real_layer_pred, real_features = discriminator(gt_layer, condition)
                    d_real = real_validity.mean()
                    d_real_loss -= d_real
                    d_layer_loss_real += criterion_layer(real_layer_pred.view(batch_size, -1), layer_label)
                
                # Train with fake samples
                for i, pred_layer in enumerate(pred_layers):
                    condition = layer1 if i == 0 else gt_layers[i-1]
                    fake_validity, _, _ = discriminator(pred_layer.detach(), condition)
                    d_fake = fake_validity.mean()
                    d_fake_loss += d_fake

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / len(gt_layers)
                d_loss += lambda_layer * d_layer_loss_real / len(gt_layers)
                
                d_loss.backward()
                optimizer_D.step()

                # Clamp discriminator weights
                for p in discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

                # --- Train generator ---
                optimizer_G.zero_grad()
                pred_layers = generator(
                    layer1=layer1, 
                    gt_layer2=all_layers[:, 1],
                    gt_layer3=all_layers[:, 2],
                    gt_layer4=all_layers[:, 3], 
                    gt_layer5=all_layers[:, 4],
                    teacher_forcing=True
                )
                
                g_loss = 0
                fm_loss = 0
                g_rec_loss = 0
                g_layer_loss = 0
                g_tv_loss = 0
                
                for i, (pred_layer, gt_layer, layer_label) in enumerate(zip(pred_layers, gt_layers, layer_labels)):
                    condition = layer1 if i == 0 else gt_layers[i-1]
                    
                    # Adversarial loss (matching UNet implementation)
                    fake_validity, fake_layer_pred, fake_features = discriminator(pred_layer, condition)
                    g_adversarial = -fake_validity.mean()
                    
                    # Layer classification loss
                    g_layer_loss += criterion_layer(fake_layer_pred.view(batch_size, -1), layer_label)
                    
                    # Feature matching loss
                    _, _, real_features = discriminator(gt_layer, condition)
                    fm_loss += criterion_l1(fake_features, real_features.detach())
                    
                    # Reconstruction loss with residual regularization
                    g_rec_loss += criterion_mse(pred_layer, gt_layer)
                    g_tv_loss += total_variation_loss(pred_layer)
                    g_rec_loss += color_histogram_loss(pred_layer, gt_layer)
                    
                    # Residual regularization (sparsity)
                    prev_layer = layer1 if i == 0 else gt_layers[i-1]
                    g_rec_loss += residual_reg_weight * criterion_l1(pred_layer - prev_layer, torch.zeros_like(pred_layer))
                
                # Combine all losses (exactly as in UNet)
                g_loss = (
                    lambda_adv * g_adversarial + 
                    lambda_rec * g_rec_loss / len(gt_layers) + 
                    lambda_fm * fm_loss / len(gt_layers) +
                    lambda_layer * g_layer_loss / len(gt_layers) + 
                    lambda_tv * g_tv_loss / len(gt_layers)
                )
                
                g_loss.backward()
                optimizer_G.step()
                
                running_g_loss += g_loss.item()
                running_d_loss += d_loss.item()


        avg_g_loss = running_g_loss / (len(dataloader) * (num_layers - 1))
        avg_d_loss = running_d_loss / (len(dataloader) * (num_layers - 1))
        print(f"Epoch [{epoch+1}/{num_epochs}] - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")

        if scheduler_G is not None:
            scheduler_G.step()

        if avg_g_loss < min(g_losses):
            torch.save(generator.state_dict(), model_save_path)

        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        # --- Validation plotting code ---
        if epoch % 1 == 0:  # You can adjust frequency as needed
            if is_unet:
                plot_unet_validation(generator, dataloader.dataset, device, epoch)
            else:
                plot_vit_validation(generator, dataloader, device, epoch)

def plot_vit_validation(generator, dataset, device, epoch):
    generator.eval()
    with torch.no_grad():
        for batch in dataset:
            layer_tensor, _ = batch
            layer_tensor = layer_tensor.to(device)
            layer1 = layer_tensor[:, 0]
            gt_layer2 = layer_tensor[:, 1]
            gt_layer3 = layer_tensor[:, 2]
            gt_layer4 = layer_tensor[:, 3]
            gt_layer5 = layer_tensor[:, 4]
            
            # During evaluation, you can use teacher forcing or sequential prediction.
            # Here, we use teacher forcing for predictor3.
            pred_layer2, pred_layer3, pred_layer4, pred_layer5, pred_layer6 = generator(
                layer1=layer1, 
                gt_layer2=gt_layer2, 
                gt_layer3=gt_layer3,
                gt_layer4=gt_layer4,
                gt_layer5=gt_layer5,
                teacher_forcing=False)
            
            def to_np(t): return t[0].permute(1, 2, 0).cpu().numpy()
            
            num_layers = 6
            fig, axs = plt.subplots(1, num_layers, figsize=(15, 5))

            pred_layers = [pred_layer2, pred_layer3, pred_layer4, pred_layer5, pred_layer6]
            fig.suptitle("Generated Layers for ViT-model", fontsize=32)
            axs[0].imshow(to_np(layer1))
            axs[0].axis('off')
            axs[0].set_title("Layer 0")
            for i, layer in enumerate(pred_layers):
                img = to_np(layer)
                axs[i+1].imshow(img)
                axs[i+1].axis('off')
                axs[i+1].set_title(f"Layer {i+1}")

            plt.tight_layout()
            if epoch % 5 == 0:
                plt.savefig(f'../figures/ViT_generated_layers_{epoch}.png')
    plt.close()
    generator.train()

def plot_unet_validation(generator, dataset, device, epoch):

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
        noise = torch.randn_like(current_layer) * 0.1
        noisy_input = current_layer + noise

        with torch.no_grad():
            predicted_residual = generator(
            sample=noisy_input,
            timestep=torch.zeros(1, device=device, dtype=torch.long),
            ).sample

            next_layer = current_layer + predicted_residual
            generated_layers.append(next_layer.squeeze(0).cpu())
            current_layer = next_layer

    # Plot both original and generated sequences
    fig, (ax1, ax2) = plt.subplots(2, num_layers, figsize=(20, 8))

    for i in range(num_layers):
        original = layer_sequence[0, i].cpu().permute(1, 2, 0).clamp(0, 1)
        ax1[i].imshow(original)
        ax1[i].axis('off')
        ax1[i].set_title(f"Original {i}")

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
        residual = (residual - residual.min()) / (residual.max() - residual.min())
        axs[i].imshow(residual)
        axs[i].axis('off')
        axs[i].set_title(f"Residual {i}")

        plt.suptitle("Generated Residuals Between Layers")
        plt.tight_layout()
        plt.savefig('residuals_unet.png')
        plt.close()

    generator.train()
