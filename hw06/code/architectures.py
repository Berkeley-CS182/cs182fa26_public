import torch
import torch.nn as nn
import torchvision

class BasicConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class ResNet18(nn.Module):
    """An untrained ResNet-18; adaptive pooling also accepts 32x32 CIFAR inputs."""
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(num_ftrs, 10)
    def forward(self, x):
        return self.backbone(x)


class MLP(nn.Module):
    """MLP for 3x32x32 images; num_layers counts all hidden linear layers."""
    def __init__(self, num_layers=2, size=256, num_classes=10):
        super().__init__()
        if num_layers < 1 or size < 1:
            raise ValueError("num_layers and size must be positive")
        self.fc1 = nn.Linear(3072, size)
        self.hidden = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers - 1)])
        self.out = nn.Linear(size, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        for layer in self.hidden:
            x = layer(x)
            x = self.relu(x)
        x = self.out(x)
        return x
        
        
