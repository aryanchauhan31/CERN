import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def check_h5py_schema(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            def print_structure(name, obj):
                print(f"{name}: {type(obj)}")
                if isinstance(obj, h5py.Dataset):
                    print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}")

            f.visititems(print_structure)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace 'file_path' with the actual path to your HDF5 file
file_path = '/content/SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5'
check_h5py_schema(file_path)

# file_path = '/content/SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5'
# check_h5py_schema(file_path)

class ParticleDataset(Dataset):
    def __init__(self, file_path, particle_type):
        """
        Args:
            file_path: Path to the h5 file
            particle_type: 0 for photons, 1 for electrons
        """
        with h5py.File(file_path, 'r') as f:
            self.data = f['X'][:]  # Shape: (N, 32, 32, 2)
        
        # Normalize data
        self.data = self.data / np.max(self.data)
        self.labels = np.full(len(self.data), particle_type, dtype=np.int64)
        
        print(f"Loaded {len(self.data)} samples of type {particle_type}")
        print(f"Data shape: {self.data.shape}, Data type: {self.data.dtype}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]  # Shape: (32, 32, 2)
        sample = torch.tensor(sample, dtype=torch.float32).permute(2, 0, 1)  # Shape: (2, 32, 32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if idx < 3:
            print(f"Sample {idx} shape: {sample.shape}, Label: {label.item()}")
        return sample, label



class SEBlock(nn.Module):
  def __init__(self, channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        
        # Squeeze operation - global average pooling
        y = self.avg_pool(x).view(batch_size, channels)
        
        # Excitation operation - adaptive recalibration
        y = self.fc(y).view(batch_size, channels, 1, 1)
        
        # Scale the input tensor
        return x * y.expand_as(x)

class SEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction_ratio=16):
        super(SEResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # SE block
        self.se = SEBlock(out_channels, reduction_ratio)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
     def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply SE block
        out = self.se(out)
        
        # Add shortcut connection
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class SEResNet15(nn.Module):
    def __init__(self, num_classes=2):
        super(SEResNet15, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual layers with SE blocks
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        
        # First block may downsample
        layers.append(SEResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        # Remaining blocks maintain dimensions
        for _ in range(1, num_blocks):
            layers.append(SEResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def train_model(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Move to device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Debug first batch of first epoch
        if epoch == 0 and batch_idx == 0:
            print(f"\nTraining - First batch:")
            print(f"Input batch shape: {inputs.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Input device: {inputs.device}, Model device: {next(model.parameters()).device}")
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Check if outputs is None or not a tensor
        if outputs is None or not torch.is_tensor(outputs):
            print(f"ERROR: Model output is {type(outputs)}")
            continue
            
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print progress for first epoch
        if epoch == 0 and batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc, all_preds, all_labels


from torchvision import transforms
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def main():
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    photon_path = '/content/SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5'  # Update with actual path
    electron_path = '/content/SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5'  # Update with actual path
    
    print("\nLoading datasets...")
    photon_dataset = ParticleDataset(photon_path, particle_type=0)
    electron_dataset = ParticleDataset(electron_path, particle_type=1)
    
    # Combine datasets
    combined_dataset = torch.utils.data.ConcatDataset([photon_dataset, electron_dataset])
    print(f"Combined dataset size: {len(combined_dataset)}")
    
    # Split into train and test sets (80/20)
    train_size = int(0.8 * len(combined_dataset))
    test_size = len(combined_dataset) - train_size
    train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])
    train_dataset.dataset.transform = train_transforms

    test_transforms = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset.dataset.transform = test_transforms

    print(f"Train set: {len(train_dataset)}, Test set: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create model
    print("\nCreating model...")
    # 
    model = SEResNet15(num_classes=2)
    
    # Move model to device
    model = model.to(device)
    print(f"Model device: {next(model.parameters()).device}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # Training loop
    print("\nStarting training...")
    num_epochs = 15
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device, epoch)
        
        # Evaluate
        test_loss, test_acc, all_preds, all_labels = evaluate_model(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(test_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if epoch == 0 or test_acc > max(test_accs[:-1]):
            torch.save(model.state_dict(), 'best_se_resnet15_model.pth')
            print(f"New best model saved with accuracy: {test_acc:.2f}%")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load('best_se_resnet15_model.pth'))
    _, final_acc, all_preds, all_labels = evaluate_model(model, test_loader, criterion, device)
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Photon', 'Electron'], 
                yticklabels=['Photon', 'Electron'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - SE-ResNet-15')
    plt.savefig('se_confusion_matrix.png')
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Photon', 'Electron']))
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves - SE-ResNet-15')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves - SE-ResNet-15')
    
    plt.tight_layout()
    plt.savefig('se_training_curves.png')
    
    # Save model
    torch.save(model.state_dict(), 'se_resnet15_model.pth')
    print("\nTraining complete. SE-ResNet-15 model saved.")

if __name__ == "__main__":
    main()
