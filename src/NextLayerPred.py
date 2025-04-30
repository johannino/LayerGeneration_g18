import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from CharacterLoader import CharacterLayerLoader
import matplotlib.pyplot as plt
from Discriminator import LayerDiscriminator
from Loss_functions import color_histogram_loss

# ---------------------------
# Vision Transformer for Residual Regression
# ---------------------------
class VisionTransformerForRegression(nn.Module):
    """
    A simple Vision Transformer that takes an input layer (e.g., layer1 or layer2)
    and predicts a residual such that: predicted_layer = input_layer + residual.
    This encourages the network to preserve the base and only "add things."
    """
    def __init__(
        self,
        image_size=100,
        patch_size=10,
        in_channels=3,
        embed_dim=256,
        nhead=4,
        num_layers=5, # Number of Transformer encoder layers
    ):
        super().__init__()

        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Number of patches (e.g., (100/10)*(100/10) = 100)
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)

        # Linear projection for each patch: from (in_channels*patch_size^2) -> embed_dim
        self.patch_embed = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

        # Positional embeddings for each patch (learnable)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Linear projection back to image patches (for predicting the residual)
        self.patch_unembed = nn.Linear(embed_dim, in_channels * patch_size * patch_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embedding, std=0.02)
        # Initialize the final projection layer to output near-zero residuals.
        nn.init.constant_(self.patch_unembed.weight, 0)
        nn.init.constant_(self.patch_unembed.bias, 0)

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, in_channels, H, W] (input layer).
        Returns: final_output = x + predicted_residual, where predicted_residual is computed via the transformer.
        """
        x_orig = x  # Save input for skip connection

        # 1) Patchify input
        patches = self._patchify(x)  # [B, num_patches, patch_dim]
        
        # 2) Project patches to embedding space
        x = self.patch_embed(patches)  # [B, num_patches, embed_dim]

        # 3) Add positional embeddings
        x = x + self.pos_embedding[:, :self.num_patches, :]

        # 4) Permute to [num_patches, B, embed_dim] for transformer
        x = x.permute(1, 0, 2)

        # 5) Transformer encoder
        x = self.transformer_encoder(x)  # [num_patches, B, embed_dim]

        # 6) Permute back to [B, num_patches, embed_dim]
        x = x.permute(1, 0, 2)

        # 7) Map embeddings back to patch pixels (predict residual)
        residual_patches = self.patch_unembed(x)  # [B, num_patches, patch_dim]

        # 8) Unpatchify to reconstruct the residual image
        predicted_residual = self._unpatchify(residual_patches)  # [B, in_channels, H, W]

        # 9) Compute final output as input + residual
        final_output = x_orig + predicted_residual

        # 10) Clamp final output to [0,1]
        final_output = torch.clamp(final_output, 0.0, 1.0)
        return final_output

    def _patchify(self, imgs):
        b, c, h, w = imgs.shape
        p = self.patch_size
        imgs = imgs.reshape(b, c, h // p, p, w // p, p)
        imgs = imgs.permute(0, 2, 4, 3, 5, 1).contiguous()
        patches = imgs.view(b, (h // p) * (w // p), p * p * c)
        return patches

    def _unpatchify(self, patches):
        b, npatch, _ = patches.shape
        p = self.patch_size
        c = self.in_channels
        h = self.image_size
        w = self.image_size

        patches = patches.view(b, h // p, w // p, p * p * c)
        patches = patches.view(b, h // p, w // p, p, p, c)
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        imgs = patches.view(b, c, h, w)
        return imgs

# ---------------------------
# Composite Model for Multi-Layer Prediction
# ---------------------------
class MultiLayerPredictor(nn.Module):
    """
    A composite model that predicts multiple extra layers sequentially.
    - predictor2: Predicts layer2 from layer1.
    - predictor3: Predicts layer3 from layer2.
    During training, teacher forcing is used: the ground truth layer2 is fed into predictor3.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.predictor2 = VisionTransformerForRegression(**kwargs)
        self.predictor3 = VisionTransformerForRegression(**kwargs)
        self.predictor4 = VisionTransformerForRegression(**kwargs)
        self.predictor5 = VisionTransformerForRegression(**kwargs)
        self.predictor6 = VisionTransformerForRegression(**kwargs)
    
    def forward(self, layer1, gt_layer2=None, gt_layer3=None, gt_layer4=None, gt_layer5=None,teacher_forcing=True):
        # Predict layer2 from layer1.
        pred_layer2 = self.predictor2(layer1)
        # For predictor3, use teacher forcing if available.
        # Predict layer3 using either ground truth or predicted layer2
        input_for_layer3 = gt_layer2 if teacher_forcing and gt_layer2 is not None else pred_layer2
        pred_layer3 = self.predictor3(input_for_layer3)

        # Predict layer4 using either ground truth or predicted layer3
        input_for_layer4 = gt_layer3 if teacher_forcing and gt_layer3 is not None else pred_layer3
        pred_layer4 = self.predictor4(input_for_layer4)

        # Predict layer 5
        input_for_layer5 = gt_layer4 if teacher_forcing and gt_layer4 is not None else pred_layer4
        pred_layer5 = self.predictor5(input_for_layer5)
        
        # Predict layer 5
        input_for_layer6 = gt_layer5 if teacher_forcing and gt_layer5 is not None else pred_layer5
        pred_layer6 = self.predictor6(input_for_layer6)

        return pred_layer2, pred_layer3, pred_layer4, pred_layer5, pred_layer6

# ---------------------------
# Driver Code
# ---------------------------
if __name__ == "__main__":
    # 1. Create the dataset and dataloader.
    # Note: Maybe we can normalize images in CharacterLayerLoader.
    # Also, we are starting from shirt layer and not base (need to check datalaoder)
    batch_size = 32
    data_folder = "../data/"
    dataset = CharacterLayerLoader(data_folder=data_folder, resolution=(100, 100))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Number of batches: {len(dataloader)}")

    # 2. Instantiate the multi-layer predictor.
    # We'll use the same configuration for both predictor modules.
    model = MultiLayerPredictor(
        image_size=100,
        patch_size=10,
        in_channels=3,
        embed_dim=256,
        nhead=4,
        num_layers=4
    ).to(device)

    discriminator = LayerDiscriminator(
        image_size=100,
        in_channels=3
    ).to(device)

    # 3. Define loss and optimizer.
    # We use MSE loss plus an L1 penalty on the residuals to encourage minimal changes.
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_bce = nn.BCELoss()
    criterion_layer = nn.CrossEntropyLoss()

    lambda_adv = 1.0      # Adversarial loss weight
    lambda_rec = 10.0     # Reconstruction loss weight
    lambda_fm = 10.0      # Feature matching loss weight
    lambda_layer = 2.0    # Layer classification loss weight

    optimizer_G= optim.Adam(model.parameters(), lr=1e-4)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=200)
    residual_reg_weight = 0.1  # Weight for residual regularization.
    adversarial_weight = 0.5

    real_label, fake_label = 1.0, 0.0
    # 4. Training loop.
    # We assume the dataset provides a tensor of shape [batch, num_layers, 3, 100, 100].
    # For multi-layer prediction, we use:
    #   - layer1 as input,
    #   - layer2 as the target for predictor2,
    #   - layer3 as the target for predictor3.
    num_epochs = 25
    model.train()
    discriminator.train()

    for epoch in range(num_epochs):
        running_g_loss = 0.0
        running_d_loss = 0.0
        for batch in dataloader:
            # layer_tensor shape: [batch_size, 5, 3, 100, 100] (if 5 layers per character)
            layer_tensor, _ = batch
            layer_tensor = layer_tensor.to(device)
            layer1 = layer_tensor[:, 0]   # Base layer.

            gt_layers = [layer_tensor[:, i] for i in range(1, 6)]  # Layers 2-6

            real_target = torch.full((batch_size, 1, 3, 3), real_label, device=device)
            fake_target = torch.full((batch_size, 1, 3, 3), fake_label, device=device)
            layer_labels = [torch.full((batch_size,), i, dtype=torch.long, device=device) for i in range(5)]

            # 4.1 Train the discriminator.
            d_real_loss = 0
            d_layer_loss_real = 0
            optimizer_D.zero_grad()
            
            pred_layers = model(
                layer1=layer1, 
                gt_layer2=layer_tensor[:, 1],
                gt_layer3=layer_tensor[:, 2],
                gt_layer4=layer_tensor[:, 3], 
                gt_layer5=layer_tensor[:, 4],
                teacher_forcing=True)

            d_real_loss = 0
            d_layer_loss_real = 0

            # Train with real samples
            for i, (gt_layer, layer_label) in enumerate(zip(gt_layers, layer_labels)):
                condition = layer1 if i == 0 else gt_layers[i-1]
                
                real_validity, real_layer_pred, _ = discriminator(gt_layer, condition)
                d_real_loss += criterion_bce(real_validity, real_target)
                d_layer_loss_real += criterion_layer(real_layer_pred.view(batch_size, -1), layer_label)
            
            # Train with fake samples
            d_fake_loss = 0
            for i, pred_layer in enumerate(pred_layers):
                condition = layer1 if i == 0 else gt_layers[i-1]
                fake_validity, _, _ = discriminator(pred_layer.detach(), condition)
                d_fake_loss += criterion_bce(fake_validity, fake_target)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / len(gt_layers) + d_layer_loss_real / len(gt_layers)
            d_loss.backward()
            optimizer_D.step()

            # 4.2 Train the generator.
            optimizer_G.zero_grad()
            # Use teacher forcing for the second stage.
            pred_layers= model(
                layer1=layer1, 
                gt_layer2=layer_tensor[:, 1],
                gt_layer3=layer_tensor[:, 2],
                gt_layer4=layer_tensor[:, 3], 
                gt_layer5=layer_tensor[:, 4],
                teacher_forcing=True
                )
            
            g_loss = 0
            fm_loss = 0
            g_rec_loss = 0
            g_layer_loss = 0
            
            for i, (pred_layer, gt_layer, layer_label) in enumerate(zip(pred_layers, gt_layers, layer_labels)):
                condition = layer1 if i == 0 else gt_layers[i-1]
                
                # Adversarial loss
                fake_validity, fake_layer_pred, fake_features = discriminator(pred_layer, condition)
                g_adversarial = criterion_bce(fake_validity, real_target)
                
                # Layer classification loss (generator should produce layers that match expected layer type)
                g_layer_loss += criterion_layer(fake_layer_pred.view(batch_size, -1), layer_label)
                
                # Feature matching loss
                _, _, real_features = discriminator(gt_layer, condition)
                fm_loss += criterion_l1(fake_features, real_features.detach())
                
                # Reconstruction loss
                g_rec_loss += criterion_mse(pred_layer, gt_layer)

                # color_histogram_loss
                g_rec_loss += color_histogram_loss(pred_layer, gt_layer)
                
                # Add L1 loss to enforce sparsity
                prev_layer = layer1 if i == 0 else gt_layers[i-1]
                g_rec_loss += 0.1 * criterion_l1(pred_layer - prev_layer, torch.zeros_like(pred_layer))
                
            # Combine all losses
            g_loss = (lambda_adv * g_adversarial + 
                      lambda_rec * g_rec_loss / len(gt_layers) + 
                      lambda_fm * fm_loss / len(gt_layers) +
                      lambda_layer * g_layer_loss / len(gt_layers))
            
            g_loss.backward()
            optimizer_G.step()
            
            running_g_loss += g_loss.item()
            running_d_loss += d_loss.item()
        
        avg_g_loss = running_g_loss / len(dataloader)
        avg_d_loss = running_d_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
        scheduler_G.step()
        
        # Save model checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"../Models/vit_model.pth")
        

    # 5. Evaluation and save predictions.
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            layer_tensor, _ = batch
            layer_tensor = layer_tensor.to(device)
            layer1 = layer_tensor[:, 0]
            gt_layer2 = layer_tensor[:, 1]
            gt_layer3 = layer_tensor[:, 2]
            gt_layer4 = layer_tensor[:, 3]
            gt_layer5 = layer_tensor[:, 4]
            gt_layer6 = layer_tensor[:, 5]
            
            # During evaluation, you can use teacher forcing or sequential prediction.
            # Here, we use teacher forcing for predictor3.
            pred_layer2, pred_layer3, pred_layer4, pred_layer5, pred_layer6 = model(
                layer1=layer1, 
                gt_layer2=gt_layer2, 
                gt_layer3=gt_layer3,
                gt_layer4=gt_layer4,
                gt_layer5=gt_layer5,
                teacher_forcing=False)
            
            def to_np(t): return t[0].permute(1, 2, 0).cpu().numpy()
            
            fig, axes = plt.subplots(2, 6, figsize=(14, 8))
            axes[0, 0].imshow(to_np(layer1))
            axes[0, 0].set_title("Input Layer (Layer 1)")
            axes[0, 0].axis("off")

            axes[1 ,0].axis("off")

            axes[0, 1].imshow(to_np(gt_layer2))
            axes[0, 1].set_title("Ground Truth Layer 2")
            axes[0, 1].axis("off")

            axes[1, 1].imshow(to_np(pred_layer2))
            axes[1, 1].set_title("Predicted Layer 2")
            axes[1, 1].axis("off")

            axes[0, 2].imshow(to_np(gt_layer3))
            axes[0, 2].set_title("Ground Truth Layer 3")
            axes[0, 2].axis("off")

            axes[1, 2].imshow(to_np(pred_layer3))
            axes[1, 2].set_title("Predicted Layer 3")
            axes[1, 2].axis("off")
            
            axes[0, 3].imshow(to_np(gt_layer4))
            axes[0, 3].set_title("Ground Truth Layer 4")
            axes[0, 3].axis("off")

            axes[1, 3].imshow(to_np(pred_layer4))
            axes[1, 3].set_title("Predicted Layer 4")
            axes[1, 3].axis("off")

            axes[0, 4].imshow(to_np(gt_layer5))
            axes[0, 4].set_title("Ground Truth Layer 5")
            axes[0, 4].axis("off")

            axes[1, 4].imshow(to_np(pred_layer5))
            axes[1, 4].set_title("Predicted Layer 5")
            axes[1, 4].axis("off")

            axes[0, 5].imshow(to_np(gt_layer6))
            axes[0, 5].set_title("Ground Truth Layer 6")
            axes[0, 5].axis("off")

            axes[1, 5].imshow(to_np(pred_layer6))
            axes[1, 5].set_title("Predicted Layer 6")
            axes[1, 5].axis("off")

            axes[1 ,0].axis("off")

            plt.tight_layout()
            plt.savefig("../figures/predictions_multilayer.png")
            plt.show()
            break  # Only evaluate one batch
