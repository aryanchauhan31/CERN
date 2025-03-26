import torch 
import torch.nn as nn
import torch.nn.functional as F

class Resnet18(nn.Module):
  def __init__(self, num_classes=1000):
    super(Resnet18, self).__init__()
    self.conv1 = nn.Conv2d()
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inPlace=True)
    self.maxpool = nn.MaxPool2d()

    self.layer1 = self._make_layer(64,64,2)
    self.layer2 = self._make_layer(64,128,2, stride=2)
    self.layer3 = self._make_layer(128,256,2, stride=2)
    self.layer4 = self._make_layer(256,512,2,stride=2)

    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(512, num_classes)

  def _make_layer(self, in_channels, out_channels, blocks, stride=1):
    layers = []
    layers.append(self._block(in_channels,out_channels, stride))
    for _ in range(1,blocks):
      layers.append(self._block(out_channels,out_channels))
      return nn.Seequential(*layers)
    
  def _block(self,in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inPlace=True)
    )
  
  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = self.flatten(x,1)
    x = self.fc(x)

    return x

class ResNet18BackBone(nn.Module):
  def __init__(self):
    super(ResNet18BackBone, self).__init__()
    self.fc = nn.Identity()

  def forward(self,x):
    x = nn.conv1(x)
    x = nn.bn1(x)
    x = nn.relu(x)
    x = nn.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = self.flatten(x,1)

    return x

class SimCLR(nn.Module):
  def __init__(self, feature_dim=128):
    super(SimCLR, self).__init__()
  
    self.encoder = ResNet18BackBone()
    self.projection_head = nn.Sequential(
        nn.Linear(512,512),
        nn.ReLU(inPlace=True),
        nn.Linear(512,feature_dim)
    )

  def forward(self,x):
    h = self.encoder(x)
    z = self.projection_head(h)
    return F.normalize(z, dim=1)
  
  def get_encoder(self):
    return self.encoder
