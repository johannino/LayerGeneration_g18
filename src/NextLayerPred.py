import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from CharacterLoader import CharacterLayerLoader
import matplotlib.pyplot as plt
from Discriminator import LayerDiscriminator
from Loss_functions import color_histogram_loss, total_variation_loss, gradient_penalty

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
        image_size=256,
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
    # 1. Setup
    batch_size = 16
    data_folder = "../data/"
    dataset = CharacterLayerLoader(data_folder=data_folder, resolution=(256, 256))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Number of batches: {len(dataloader)}")

    # 2. Initialize models
    model = MultiLayerPredictor(
        image_size=256,
        patch_size=16,
        in_channels=3,
        embed_dim=256,
        nhead=4,
        num_layers=4
    ).to(device)

    discriminator = LayerDiscriminator(
        image_size=256,
        in_channels=3
    ).to(device)

    # 3. Setup optimizers
    optimizer_G = optim.Adam(model.parameters(), lr=1e-4)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=200)

    # 4. Import and use GAN training function
    from GAN_train import train_gan

    train_gan(
        generator=model,
        discriminator=discriminator,
        dataloader=dataloader,
        device=device,
        num_epochs=30,
        lambda_adv=1.0,
        lambda_rec=10.0,
        lambda_fm=10.0,
        lambda_layer=2.0,
        lambda_tv=1,
        residual_reg_weight=0.1,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        scheduler_G=scheduler_G,
        model_save_path="../models/vit_model.pth"
    )
        
