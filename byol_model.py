"""
BYOL (Bootstrap Your Own Latent) implementation using LightlySSL framework.
"""

import copy
import torch

from torch import nn
from torchvision import models
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLProjectionHead, BYOLPredictionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum


class BYOL(nn.Module):
    """
    BYOL model implementation for self-supervised learning.

    Args:
        backbone: The backbone network (e.g., ResNet)
        input_dim: Input dimension for projection head
        hidden_dim: Hidden dimension for projection head
        output_dim: Output dimension for projection head
        momentum: Momentum parameter for target network updates
    """

    def __init__(
        self, backbone, input_dim=2048, hidden_dim=4096, output_dim=256, momentum=0.996
    ):
        super().__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(input_dim, hidden_dim, output_dim)
        self.prediction_head = BYOLPredictionHead(output_dim, hidden_dim, output_dim)

        # Create momentum versions of backbone and projection head
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        # Deactivate gradients for momentum networks
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NegativeCosineSimilarity()
        self.momentum = momentum

    def forward(self, x0, x1):
        """
        Forward pass for BYOL training.

        Args:
            x0: First augmented view of the input
            x1: Second augmented view of the input

        Returns:
            Loss value
        """
        # Online network forward pass
        y0 = self.projection_head(self.backbone(x0))
        y1 = self.projection_head(self.backbone(x1))

        # Prediction heads
        p0 = self.prediction_head(y0)
        p1 = self.prediction_head(y1)

        # Target network forward pass (no gradients)
        with torch.no_grad():
            y0_m = self.projection_head_momentum(self.backbone_momentum(x0))
            y1_m = self.projection_head_momentum(self.backbone_momentum(x1))

        # Compute symmetric loss
        loss = 0.5 * (self.criterion(p0, y1_m) + self.criterion(p1, y0_m))

        return loss

    @torch.no_grad()
    def update_momentum(self):
        """Update momentum networks using exponential moving average."""
        update_momentum(self.backbone, self.backbone_momentum, m=self.momentum)
        update_momentum(
            self.projection_head, self.projection_head_momentum, m=self.momentum
        )

    def get_backbone_features(self, x):
        """
        Extract features from the backbone network.
        Useful for downstream tasks after pretraining.
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features


def create_byol_model(backbone_name="resnet18", pretrained=False, **kwargs):
    """
    Create a BYOL model with specified backbone.

    Args:
        backbone_name: Name of the backbone architecture
        pretrained: Whether to use pretrained weights for backbone
        **kwargs: Additional arguments for BYOL model

    Returns:
        BYOL model instance
    """

    # Create backbone
    if backbone_name == "resnet18":
        backbone = models.resnet18(weights=pretrained)
        # Remove the final classification layer and adaptive pooling
        backbone = nn.Sequential(
            *list(backbone.children())[:-2]
        )  # Remove avgpool and fc
        input_dim = 512
    elif backbone_name == "resnet50":
        backbone = models.resnet50(weights=pretrained)
        backbone = nn.Sequential(
            *list(backbone.children())[:-2]
        )  # Remove avgpool and fc
        input_dim = 2048
    elif backbone_name == "resnet34":
        backbone = models.resnet34(weights=pretrained)
        backbone = nn.Sequential(
            *list(backbone.children())[:-2]
        )  # Remove avgpool and fc
        input_dim = 512
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # Add adaptive average pooling to get consistent output size
    backbone = nn.Sequential(backbone, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    # Create BYOL model
    model = BYOL(backbone, input_dim=input_dim, **kwargs)

    return model
