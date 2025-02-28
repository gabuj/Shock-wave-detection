import torch
import torch.nn as nn
from torchvision import models

class UNet(nn.Module):
    def __init__(self, pretrained=True):
        super(UNet, self).__init__()
        
        # Encoder: Pre-trained ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last two layers
        
        # Decoder: Deconvolution layers to reconstruct the output to the input size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        )
        
    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return torch.sigmoid(x2)  # Output between 0 and 1

# Initialize model
model = UNet()
