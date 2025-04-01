import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

file_path = '/content/SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5'
check_h5py_schema(file_path)

# file_path = '/content/SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5'
# check_h5py_schema(file_path)


class ParticleDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path

        
        with h5py.File(file_path, 'r') as f:
            self.data = np.array(f['X'])  
            self.labels = np.array(f['y'])
        
        self.data = self.data / np.max(self.data) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = int(self.labels[idx])
        sample = torch.tensor(sample, dtype=torch.float32).permute(2, 0, 1)  
        label = torch.tensor(label, dtype=torch.long)  
        return sample, label


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        print(f"Created ResidualBlock: in={in_channels}, out={out_channels}, stride={stride}")

    def forward(self, x):

        if not torch.is_tensor(x):
            print(f"ERROR: Input to ResidualBlock is not a tensor: {type(x)}")
            return None

        print(f"ResBlock input shape: {x.shape}")

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        print(f"After first conv+bn+relu: {out.shape}")

        out = self.conv2(out)
        out = self.bn2(out)

        print(f"After second conv+bn: {out.shape}")

        shortcut_out = self.shortcut(identity)
        print(f"Shortcut output shape: {shortcut_out.shape}")

        if out.shape != shortcut_out.shape:
            print(f"ERROR: Shape mismatch - out: {out.shape}, shortcut: {shortcut_out.shape}")

        out += shortcut_out
        out = self.relu(out)

        print(f"ResBlock output shape: {out.shape}")
        return out

class ResNet15(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet15, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        print("ResNet15 model created")
        print(f"Initial conv: in=2, out=64")
        print(f"Layer1: in=64, out=64")
        print(f"Layer2: in=64, out=128, stride=2")
        print(f"Layer3: in=128, out=256, stride=2")
        print(f"Final FC: in=256, out={num_classes}")

    def _make_layer(self, out_channels, num_blocks, stride=1):
        layers = []

        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
  
        if not torch.is_tensor(x):
            print(f"ERROR: Input to ResNet15 is not a tensor: {type(x)}")
            return None

        print(f"Model input shape: {x.shape}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        print(f"After initial conv+bn+relu: {x.shape}")
        x = self.layer1(x)
        print(f"After layer1: {x.shape}")

        x = self.layer2(x)
        print(f"After layer2: {x.shape}")

        x = self.layer3(x)
        print(f"After layer3: {x.shape}")

        x = self.avgpool(x)
        print(f"After avgpool: {x.shape}")

        x = torch.flatten(x, 1)
        print(f"After flatten: {x.shape}")

        x = self.fc(x)
        print(f"Final output shape: {x.shape}")
        return x


def load_data():
  photon_dataset_path = '/content/SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5'
  electron_dataset_path = '/content/SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5'

  photon_dataset = ParticleDataset(photon_dataset_path)
  electron_dataset = ParticleDataset(electron_dataset_path)

  combined_dataset = torch.utils.data.ConcatDataset([photon_dataset, electron_dataset])
  train_size = int(0.8*len(combined_dataset))
  test_size = len(combined_dataset)-train_size
  train_dataset, test_dataset = random_split(combined_dataset, [train_size,test_size])
  return train_dataset, test_dataset


def train_model(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        if epoch == 0 and batch_idx == 0:
            print(f"\nTraining - First batch:")
            print(f"Input batch shape: {inputs.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Input device: {inputs.device}, Model device: {next(model.parameters()).device}")

        optimizer.zero_grad()
        outputs = model(inputs)
        if outputs is None or not torch.is_tensor(outputs):
            print(f"ERROR: Model output is {type(outputs)}")
            continue

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

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
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(test_loader)
    test_acc = correct / total
    return test_loss, test_acc, all_preds, all_labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    photon_path = '/content/SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5' 
    electron_path = '/content/SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5'  

    print("\nLoading datasets...")
    photon_dataset = ParticleDataset(photon_path, particle_type=0)
    electron_dataset = ParticleDataset(electron_path, particle_type=1)

    combined_dataset = torch.utils.data.ConcatDataset([photon_dataset, electron_dataset])
    print(f"Combined dataset size: {len(combined_dataset)}")

    train_size = int(0.8 * len(combined_dataset))
    test_size = len(combined_dataset) - train_size
    train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])
    print(f"Train set: {len(train_dataset)}, Test set: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    print("\nCreating model...")
    model = ResNet15(num_classes=2)

    model = model.to(device)
    print(f"Model device: {next(model.parameters()).device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    print("\nStarting training...")
    num_epochs = 10
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    torch.set_printoptions(profile="default")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device, epoch)

        test_loss, test_acc, _, _ = evaluate_model(model, test_loader, criterion, device)

        scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    print("\nFinal evaluation...")
    _, final_acc, all_preds, all_labels = evaluate_model(model, test_loader, criterion, device)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Photon', 'Electron'],
                yticklabels=['Photon', 'Electron'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Photon', 'Electron']))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')

    plt.tight_layout()
    plt.savefig('training_curves.png')

    torch.save(model.state_dict(), 'resnet15_model.pth')
    print("\nTraining complete. Model saved.")

if __name__ == "__main__":
    main()
