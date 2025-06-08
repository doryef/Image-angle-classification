import torch.nn as nn
import torchvision.models as models

class AngleClassifier(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True):
        super(AngleClassifier, self).__init__()
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        
        # Freeze the backbone if specified
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3)  # 3 classes: 0-30, 30-60, 60-90
        )

    def forward(self, x):
        return self.resnet(x)

    def unfreeze_layers(self, num_layers=0):
        """Unfreeze the last n layers of the ResNet backbone"""
        if num_layers == 0:
            return
        
        layers_to_unfreeze = list(self.resnet.named_parameters())
        for name, param in layers_to_unfreeze[-num_layers:]:
            param.requires_grad = True