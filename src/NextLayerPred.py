import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Model.CharacterLoader import CharacterLayerLoader
import matplotlib.pyplot as plt
import math

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
        num_layers=4, # Number of Transformer encoder layers
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
# Composite Model for Multi-Layer Prediction (3 layers for the moment)
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
    
    def forward(self, layer1, gt_layer2=None, teacher_forcing=True):
        # Predict layer2 from layer1.
        pred_layer2 = self.predictor2(layer1)
        # For predictor3, use teacher forcing if available.
        if teacher_forcing and gt_layer2 is not None:
            input_for_layer3 = gt_layer2
        else:
            input_for_layer3 = pred_layer2
        pred_layer3 = self.predictor3(input_for_layer3)
        return pred_layer2, pred_layer3

# ---------------------------
# Driver Code
# ---------------------------
if __name__ == "__main__":
    # 1. Create the dataset and dataloader.
    # Note: Maybe we can normalize images in CharacterLayerLoader.
    # Also, we are starting from shirt layer and not base (need to check datalaoder)
    data_folder = "../data/"
    dataset = CharacterLayerLoader(data_folder=data_folder, resolution=(100, 100))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 2. Instantiate the multi-layer predictor.
    # We'll use the same configuration for both predictor modules.
    model = MultiLayerPredictor(
        image_size=100,
        patch_size=10,
        in_channels=3,
        embed_dim=256,
        nhead=4,
        num_layers=4
    )

    # 3. Define loss and optimizer.
    # We use MSE loss plus an L1 penalty on the residuals to encourage minimal changes.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    residual_reg_weight = 0.1  # Weight for residual regularization.

    # 4. Training loop.
    # We assume the dataset provides a tensor of shape [batch, num_layers, 3, 100, 100].
    # For multi-layer prediction, we use:
    #   - layer1 as input,
    #   - layer2 as the target for predictor2,
    #   - layer3 as the target for predictor3.
    num_epochs = 100
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            # layer_tensor shape: [batch_size, 5, 3, 100, 100] (if 5 layers per character)
            layer_tensor, _ = batch
            layer1 = layer_tensor[:, 0]   # Base layer.
            gt_layer2 = layer_tensor[:, 1]  # Ground truth layer2.
            gt_layer3 = layer_tensor[:, 2]  # Ground truth layer3.
            
            optimizer.zero_grad()
            # Use teacher forcing for the second stage.
            pred_layer2, pred_layer3 = model(layer1, gt_layer2=gt_layer2, teacher_forcing=True)
            
            # Loss for predictor2.
            loss_layer2 = criterion(pred_layer2, gt_layer2)
            residual_layer2 = pred_layer2 - layer1
            loss_res2 = torch.mean(torch.abs(residual_layer2))
            
            # Loss for predictor3.
            loss_layer3 = criterion(pred_layer3, gt_layer3)
            residual_layer3 = pred_layer3 - gt_layer2  # Since predictor3 adds residual over layer2.
            loss_res3 = torch.mean(torch.abs(residual_layer3))
            
            loss = loss_layer2 + residual_reg_weight * loss_res2 + loss_layer3 + residual_reg_weight * loss_res3
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")
        scheduler.step()

    # 5. Evaluation and save predictions.
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            layer_tensor, _ = batch
            layer1 = layer_tensor[:, 0]
            gt_layer2 = layer_tensor[:, 1]
            gt_layer3 = layer_tensor[:, 2]
            
            # During evaluation, you can use teacher forcing or sequential prediction.
            # Here, we use teacher forcing for predictor3.
            pred_layer2, pred_layer3 = model(layer1, gt_layer2=gt_layer2, teacher_forcing=True)
            
            # Display and save the first example.
            input_img = layer1[0].permute(1, 2, 0).cpu().numpy()
            pred_img2 = pred_layer2[0].permute(1, 2, 0).cpu().numpy()
            gt_img2 = gt_layer2[0].permute(1, 2, 0).cpu().numpy()
            pred_img3 = pred_layer3[0].permute(1, 2, 0).cpu().numpy()
            gt_img3 = gt_layer3[0].permute(1, 2, 0).cpu().numpy()
            
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            axes[0, 0].imshow(input_img)
            axes[0, 0].set_title("Input Layer (Layer 1)")
            axes[0, 0].axis("off")
            axes[0, 1].imshow(pred_img2)
            axes[0, 1].set_title("Predicted Layer 2")
            axes[0, 1].axis("off")
            axes[0, 2].imshow(gt_img2)
            axes[0, 2].set_title("Ground Truth Layer 2")
            axes[0, 2].axis("off")
            axes[1, 0].axis("off")  # Empty
            axes[1, 1].imshow(pred_img3)
            axes[1, 1].set_title("Predicted Layer 3")
            axes[1, 1].axis("off")
            axes[1, 2].imshow(gt_img3)
            axes[1, 2].set_title("Ground Truth Layer 3")
            axes[1, 2].axis("off")
            plt.tight_layout()
            plt.savefig("predictions_multilayer.png")
            plt.show()
            break  # Only evaluate one batch
