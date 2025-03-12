import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class UNet(nn.Module):
    def __init__(self, pretrained=True):
        super(UNet, self).__init__()
        
        # Encoder: Pre-trained ResNet18
        
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify ResNet's first conv layer to accept grayscale images (1 channel), maybe change kernel size to 3 and padding to 1?
        resnet.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=3)

        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last two layers
        
        # Decoder: Deconvolution layers to reconstruct the output to input size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            #output layer, size should match:(torch.Size([1, 1, 334, 409])), 
            #keeping in mind: output_size = floor((input_size + 2 * padding - kernel_size) / stride) + 1
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        )
        
    def forward(self, x): #x is the input image
        x1 = self.encoder(x)  # Pass input through encoder (modified ResNet18)
        x2 = self.decoder(x1)  # Pass features through decoder
        # Resize output to match target labels
        x2 = F.interpolate(x2, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return torch.sigmoid(x2)  # Sigmoid activation for binary mask output