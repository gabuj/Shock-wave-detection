import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
class UNet(nn.Module):
    def __init__(self, pretrained=True):
        super(UNet, self).__init__()
        
        # Encoder: Pre-trained ResNet18
        
        # Load pre-trained VGG16 model (HED is based on VGG16)
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)


        # Extract feature layers (HED uses up to Conv5)
        self.encoder = vgg.features[:24]  # Up to the last convolutional layer before FC

        # Modify first conv layer for grayscale input (1 channel instead of 3)
        self.encoder[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        
        # Decoder: Deconvolution layers to reconstruct the output to input size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
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