import torch.nn.functional as F
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight_edge=10, weight_bg=1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight_edge = weight_edge
        self.weight_bg = weight_bg

    def forward(self, pred, target):
        """
        pred: Raw logits from the network (NOT passed through sigmoid)
        target: Binary mask (0 for background, 1 for edge pixels)
        """
        target = target.long()  # Ensure target is long type for cross-entropy

        # Create weight map: Edge pixels have higher weight
        weight_map = torch.ones_like(target, dtype=torch.float32) * self.weight_bg
        weight_map[target == 1] = self.weight_edge  # Assign weight to edge pixels

        # Compute cross-entropy loss (logits required, no sigmoid!)
        ce_loss = F.cross_entropy(pred, target, weight=weight_map, reduction='none')

        # Apply weighting
        weighted_loss = (ce_loss * weight_map).mean()
        return weighted_loss
