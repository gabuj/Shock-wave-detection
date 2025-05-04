import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

class UNet(nn.Module):
    def __init__(self, pretrained=True):
        super(UNet, self).__init__()
        
        # Encoder: Use VGG16 as the backbone
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.enc1 = nn.Sequential(*list(vgg.features)[:5])   # conv1_2 + pool
        self.enc2 = nn.Sequential(*list(vgg.features)[5:10])  # conv2_2 + pool
        self.enc3 = nn.Sequential(*list(vgg.features)[10:17]) # conv3_3 + pool
        self.enc4 = nn.Sequential(*list(vgg.features)[17:24]) # conv4_3 + pool
        self.enc5 = nn.Sequential(*list(vgg.features)[24:31]) # conv5_3 + pool
        
        # Modify encoder to accept single-channel input
        self.enc1[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        
        # Decoder with skip connections
        self.dec5 = self._decoder_block(512, 512, 256)
        self.dec4 = self._decoder_block(256, 256, 128)  # Removed extra channels for skip connection
        self.dec3 = self._decoder_block(128, 128, 64)   # Removed extra channels for skip connection
        self.dec2 = self._decoder_block(64, 64, 32)     # Removed extra channels for skip connection
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Removed extra channels for skip connection
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def _decoder_block(self, in_channels, mid_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        # Decoder with resized skip connections
        # instead of concatenating directly, resize d5 to the size of e4
        d5 = self.dec5(e5)
        
        # Invece di concatenare direttamente, ridimensioniamo d5 alla dimensione di e4
        d5_resized = F.interpolate(d5, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(d5_resized)  # Utilizzo solo d5_resized, senza concatenazione
        
        d4_resized = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(d4_resized)  # Utilizzo solo d4_resized, senza concatenazione
        
        d3_resized = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(d3_resized)  # Utilizzo solo d3_resized, senza concatenazione
        
        d2_resized = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(d2_resized)  # Utilizzo solo d2_resized, senza concatenazione
        #normalise output to [0,1]
        
        # output need to have same size as input
        if d1.shape[2:] != x.shape[2:]:
            d1 = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return d1


def combined_loss(pred, target, bce_weight=0.5, fp_weight=2.0, gamma=2.0):
    # Create weight tensor for BCE to penalize false positives more
    # Higher values where target=0 (non-shock pixels) to reduce overtracing
    weights = torch.ones_like(target)
    weights = weights + (fp_weight-1.0) * (target)  # More weight where target=1 (shockwave pixels)
    
    # Focal loss component - focuses more on hard examples
    pt = target * pred + (1 - target) * (1 - pred) # Probability of true class
    focal_weights = (1 - pt) ** gamma # Focal loss factor, if very certain, weight is low, if uncertain, weight is high
    
    # Combine focal weights with our custom weights
    final_weights = weights * focal_weights
    
    # Weighted BCE using the built-in weight parameter
    bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
    weighted_bce = (bce_loss * final_weights).mean()
    
    # Modified Dice Loss with emphasis on false negatives
    smooth = 1.0
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Standard intersection
    intersection = (pred_flat * target_flat).sum()
    
    # False positives (pred=1 where target=0)
    false_negatives = ((1 - pred_flat) * target_flat).sum()
    
    # Penalize false positives in denominator
    union = pred_flat.sum() + target_flat.sum() + (fp_weight - 1.0) * false_negatives
    
    # Modified Dice loss
    dice = 1 - (2.0 * intersection + smooth) / (union + smooth) #
    
    # Combined loss
    return bce_weight * weighted_bce + (1 - bce_weight) * dice
