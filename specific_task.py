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
      return nn.Sequential(*layers)
    
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

class NTXentLoss(nn.Module):
  def __init__(self, temperature=0.5, batch_size=256):
    super(NTXentLoss, self).__init__()
    self.temperature = temperature
    self.batch_size = batch_size
    self.mask = self._get_mask().cuda()
    self.criterion = nn.CrossEntropyLoss(reduction='sum')
    self.similarity_f = nn.CosineSimilarity(dim=2)
  
  def _get_mask(self):
    mask = torch.zeros((2 * self.batch_size, 2 * self.batch_size))
    for i in range(self.batch_size):
      mask[i, self.batch_size+i] = 1
      mask[self.batch_size+i, i] = 1
    mas = mask.fill_diagonal_(0)
    return mask
  
  def forward(self, z_i, z_j):
    representations = torch.cat([z_i,z_j],dim=0)

    similarity_matrix = torch.matmul(representations, representations.T)/self.temperature
    sim_i_j = torch.diag(similarity_matrix, self.batch_size)
    sim_j_i = torch.diag(similarity_matrix, -self.batch_size)

    positive_pairs = torch.cat([sim_i_j,sim_j_i], dim=0)
    
    labels = torch.zeros(2 * self.batch_size).to(positive_pairs.device).long()
    
    mask = self.mask.to(similarity_matrix.device)
    similarity_matrix = similarity_matrix * mask

    logits_max, _ =  torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()

    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    loss = -mean_log_prob_pos.mean()    
    return loss
    
                                      
from torchvision import transforms
class SimCLRDataTransform:
    def __init__(self, size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(0.1 * size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __call__(self, x):
      return self.transform(x), self.transform(x)

def train_simclr(model, data_loader, optimizer, epochs=100, temperature=0.5):
    criterion = NTXentLoss(temperature=temperature, batch_size=data_loader.batch_size)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for images, _ in data_loader:  # Discard labels
            # Get the two augmented views
            x_i = images[0].cuda()
            x_j = images[1].cuda()
            
            # Forward pass
            z_i = model(x_i)
            z_j = model(x_j)
            
            # Compute loss
            loss = criterion(z_i, z_j)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch statistics
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(data_loader):.4f}")
    
    return model

class ClassificationModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super(ClassificationModel, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

class RegressionModel(nn.Module):
    def __init__(self, encoder, output_dim=1):
        super(RegressionModel, self).__init__()
        self.encoder = encoder
        self.regressor = nn.Linear(512, output_dim)
        
    def forward(self, x):
        features = self.encoder(x)
        return self.regressor(features)
