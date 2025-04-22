import torch.nn as nn
import torch

class LayerDiscriminator(nn.Module):
    def __init__(self, image_size=100, in_channels=3, num_condition_channels=3):
        super().__init__()
        
        total_channels = in_channels + num_condition_channels
        
        self.features = nn.Sequential(
            nn.Conv2d(total_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        
        self.layer_classifier = nn.Sequential(
            nn.Conv2d(512, 5, kernel_size=4, stride=1, padding=0, bias=False),
        )
    
    def forward(self, x, condition):
        x_combined = torch.cat([x, condition], dim=1)
        features = self.features(x_combined)
        validity = self.classifier(features)
        layer_type = self.layer_classifier(features)
        return validity, layer_type, features 