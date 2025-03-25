# SE-ResNet-15 for Electron/Photon Classification

This repository implements a Squeeze-and-Excitation ResNet-15 model for classifying particle detector data from CERN as either electrons or photons.

## Model Architecture

The model uses a ResNet-15 architecture enhanced with Squeeze-and-Excitation blocks to improve feature representation through attention mechanisms. This approach helps the network focus on the most discriminative features between electrons and photons.

## Results

After training for 15 epochs, the model achieved:
- Final test accuracy: ~73.3%
- Balanced performance across both classes

### Performance Visualization

#### Training and Validation Metrics
![Training Curves](https://pplx-res.cloudinary.com/image/upload/v1742938969/user_uploads/pGusion Matrix)

![Confusion Matrix](https://pplx-res.cloudinary.com/image/upload/v1742938994/user_uploads/QznYCkVwZSGzVkY/Screeusion)

The confusion matrix shows:
- True Positives (Photon): 35,935
- False Negatives (Photon): 13,658
- True Positives (Electron): 37,083
- False Negatives (Electron): 12,924

## Dataset

The model was trained on CERN detector data consisting of:
- 32×32 matrices with two channels (hit energy and time)
- Approximately 249,000 samples for each class
- 80/20 train/test split

## Implementation Details

- **Attention Mechanism**: Squeeze-and-Excitation blocks for channel-wise feature recalibration
- **Data Augmentation**: Random flips, rotations, and erasing to improve generalization
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Weight decay to prevent overfitting

## Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/electron-photon-classification.git
cd electron-photon-classification

# Install dependencies
pip install -r requirements.txt

# Run training
python CommonTask1.py
