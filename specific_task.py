import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import h5py
from torch.utils.data import DataLoader
import numpy as np

def explore_h5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        print(f"Top-level keys: {list(f.keys())}")

        def explore_group(group, prefix=""):
            for key in group.keys():
                item = group[key]
                path = f"{prefix}/{key}" if prefix else key

                if isinstance(item, h5py.Group):
                    print(f"GROUP: {path}")
                    explore_group(item, path)
                elif isinstance(item, h5py.Dataset):
                    shape = item.shape
                    dtype = item.dtype

                    print(f"DATASET: {path}")
                    print(f"  Shape: {shape}")
                    print(f"  Data type: {dtype}")

                   
                    try:
                        sample = item[0]
                        if isinstance(sample, np.ndarray):
                            print(f"  Sample shape: {sample.shape}")
                            print(f"  Sample min/max: {sample.min()}/{sample.max()}")
                        else:
                            print(f"  Sample value: {sample}")
                    except Exception as e:
                        print(f"  Error sampling data: {e}")

                    if len(item.attrs) > 0:
                        print(f"  Attributes: {list(item.attrs.keys())}")

                    print("-" * 40)

        explore_group(f)


explore_h5_file('Dataset_Specific_Unlabelled.h5')

class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform

        with h5py.File(file_path, 'r') as f:
            
            self.data_key = 'jet'
            self.length = len(f[self.data_key])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
       
        with h5py.File(self.file_path, 'r') as f:
        
            data = f[self.data_key][index]

       
        if data.shape[2] == 8:
           
            rgb_data = data[:, :, :3]
        else:
            rgb_data = data

        
        if rgb_data.max() > 1.0:
            rgb_data = rgb_data / 255.0

       
        if self.transform:
            if isinstance(self.transform, SimCLRDataTransform):
                view1 = self.transform(rgb_data)
                view2 = self.transform(rgb_data)
                return (view1, view2), -1  
            else:
                transformed_data = self.transform(rgb_data)
                return transformed_data, -1

      
        tensor_data = torch.tensor(rgb_data, dtype=torch.float32).permute(2, 0, 1)  
        return tensor_data, -1  


class Resnet18Backbone(nn.Module):
    def __init__(self, num_classes=1000, backbone_only=False):
        super(Resnet18Backbone, self).__init__()
        self.backbone_only = backbone_only

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if not backbone_only:
            self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(self._block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(self._block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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
        x = torch.flatten(x, 1)

        if not self.backbone_only:
            x = self.fc(x)

        return x

class SimCLR(nn.Module):
  def __init__(self, feature_dim=128):
    super(SimCLR, self).__init__()

    self.encoder = Resnet18Backbone(backbone_only=True)
    self.projection_head = nn.Sequential(
    nn.Linear(512, 1024),
    nn.ReLU(inplace=True), 
    nn.Linear(1024, feature_dim)
  )

  def forward(self,x):
    h = self.encoder(x)
    z = self.projection_head(h)
    return F.normalize(z, dim=1)

  def get_encoder(self):
    return self.encoder


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.15):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def _get_mask(self, batch_size):
        mask = torch.zeros((2 * batch_size, 2 * batch_size))
        for i in range(batch_size):
            mask[i, batch_size + i] = 1
            mask[batch_size + i, i] = 1
        mask = mask.fill_diagonal_(0)
        return mask

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]

        mask = self._get_mask(batch_size).to(z_i.device)

        representations = torch.cat([z_i, z_j], dim=0)

        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        sim_i_j = torch.diag(similarity_matrix, batch_size)
        sim_j_i = torch.diag(similarity_matrix, -batch_size)

        positive_pairs = torch.cat([sim_i_j, sim_j_i], dim=0)

        labels = torch.zeros(2 * batch_size).to(positive_pairs.device).long()

        similarity_matrix = similarity_matrix * mask

        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos.mean()
        return loss


 
                                      
# from torchvision import transforms
# class SimCLRDataTransform:
#     def __init__(self, size=224):
#         self.transform = transforms.Compose([
#             transforms.RandomResizedCrop(size=size),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomApply([
#                 transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
#             ], p=0.8),
#             transforms.RandomGrayscale(p=0.2),
#             transforms.GaussianBlur(kernel_size=int(0.1 * size)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#     def __call__(self, x):
#       return self.transform(x), self.transform(x)

from PIL import Image

class SimCLRDataTransform:
    def __init__(self, size=224):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

import math
# def get_lr(epoch, warmup_epochs=7, max_epochs=35, initial_lr=1e-4, base_lr=1e-3):
#     if epoch < warmup_epochs:
#         return initial_lr + (base_lr - initial_lr) * epoch / warmup_epochs
#     else:
#         return base_lr * 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
def get_lr(epoch, warmup_epochs=5, max_epochs=35, initial_lr=1e-5, base_lr=5e-3, min_lr=1e-4):
    if epoch < warmup_epochs:
        
        return initial_lr + (base_lr - initial_lr) * epoch / warmup_epochs
    else:
       
        cosine_factor = 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
        return min_lr + (base_lr - min_lr) * cosine_factor
      

from torch.cuda.amp import autocast, GradScaler
def train_simclr(model, data_loader, optimizer, epochs=100, device='cuda'):
    criterion = NTXentLoss(temperature=0.20)
    model = model.to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    scaler = GradScaler()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        current_lr = get_lr(epoch, warmup_epochs=5, max_epochs=epochs, initial_lr=1e-5, base_lr=5e-3)
        for param_group in optimizer.param_groups:
          param_group['lr'] = current_lr
    
        print(f"Epoch [{epoch+1}/{epochs}], Learning Rate: {current_lr:.6f}")
        
        for batch in data_loader:
            views, _ = batch  
            
           
            # x_i = torch.stack([view[0] for view in views]).to(device)
            # x_j = torch.stack([view[1] for view in views]).to(device)

            # z_i = model(x_i)
            # z_j = model(x_j)
            x_i = torch.stack([view[0] for view in views])  # First view
            x_j = torch.stack([view[1] for view in views])  # Second view
              
            x_i = x_i.view(-1, 3, 224, 224).to(device)
            x_j = x_j.view(-1, 3, 224, 224).to(device)
            
          
            with autocast():
                z_i = model(x_i)
                z_j = model(x_j)
                loss = criterion(z_i, z_j)


            # loss = criterion(z_i, z_j)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
       
        scheduler.step()



        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
            
        #     total_loss += loss.item()
        
      
        # scheduler.step()
        
       
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'simclr_resnet15_epoch_{epoch+1}.pt')
    
   
    torch.save(model.state_dict(), 'simclr_resnet15_final.pt')
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



def main():
    
    torch.manual_seed(42)
    np.random.seed(42)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

  
    model = SimCLR(feature_dim=128)
    model = model.to(device)
    print("Model initialized")

   
    transform = SimCLRDataTransform(size=224)
    unlabeled_dataset = H5Dataset('Dataset_Specific_Unlabelled.h5', transform=transform)

    batch_size = 128
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=batch_size,
        shuffle=True,   
        pin_memory=True,
        drop_last=True,
        num_workers=6,
        prefetch_factor=4 
    )

    print(f"Unlabeled dataset size: {len(unlabeled_dataset)}")
    print(f"Number of batches: {len(unlabeled_loader)}")


    try:
        sample_batch = next(iter(unlabeled_loader))
        print(f"Batch structure verification:")
        print(f"  Batch type: {type(sample_batch)}")
        
        views, labels = sample_batch
        print(f"  Views type: {type(views)}, Labels shape: {labels.shape}")
     
        if isinstance(views, list):
      
            if isinstance(views[0], torch.Tensor):
                print(f"  First view shape: {views[0].shape}")
                if len(views) > 1:
                    print(f"  Second view shape: {views[1].shape}")

            elif isinstance(views[0], list):
                print(f"  Views[0] is a list with {len(views[0])} elements")
                if isinstance(views[0][0], torch.Tensor):
                    print(f"  First view shape: {views[0][0].shape}")
                    if len(views[0]) > 1:
                        print(f"  Second view shape: {views[0][1].shape}")
        
        print("Batch structure looks correct for SimCLR training")
    except Exception as e:
        print(f"Error verifying batch structure: {e}")
        import traceback
        traceback.print_exc()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.06, momentum=0.9, weight_decay=5e-4)
    # from torchlars import LARS
    # optimizer = LARS(torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9))

    print("Starting SimCLR pretraining...")
    pretrained_model = train_simclr(
        model=model,
        data_loader=unlabeled_loader,
        optimizer=optimizer,
        epochs=35,  
        device=device
    )

    torch.save(pretrained_model.state_dict(), 'simclr_resnet15_pretrained.pt')
    print("SimCLR pretraining completed and model saved")
    encoder = pretrained_model.get_encoder()

if __name__ == "__main__":
  main()

