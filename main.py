import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create screenshots directory
os.makedirs('training_screenshots', exist_ok=True)
os.makedirs('test_screenshots', exist_ok=True)

class RetinaDataset(Dataset):
    def __init__(self, images, masks, transform=None, is_training=False):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.is_training = is_training
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].copy()  # Make a copy to avoid modifying original
        mask = self.masks[idx].copy()
        
        if self.transform and self.is_training:
            # Apply augmentation directly to numpy arrays
            # Convert to uint8 for OpenCV-style augmentation
            image_uint8 = (image * 255).astype(np.uint8)
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Apply augmentation
            augmented = self.transform(image=image_uint8, mask=mask_uint8)
            image = augmented['image'].astype(np.float32) / 255.0
            mask = augmented['mask'].astype(np.float32) / 255.0
        
        # Convert to tensors - images are RGB (3 channels), masks are single channel
        if len(image.shape) == 2:  # If grayscale, convert to RGB
            image = np.stack([image, image, image], axis=-1)
        
        # Convert HWC to CHW format for PyTorch
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)  # [3, H, W]
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)        # [1, H, W]
        
        return image_tensor, mask_tensor

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class MemoryEfficientUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, num_blocks=3, base_filters=32):  # Changed to 3 input channels for RGB
        super(MemoryEfficientUNet, self).__init__()
        self.num_blocks = num_blocks
        self.base_filters = base_filters
        
        # Encoder (Contracting path)
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        
        # First encoder block - now accepts 3 channels (RGB)
        self.encoder_blocks.append(ConvBlock(in_channels, base_filters))
        
        # Remaining encoder blocks
        for i in range(1, num_blocks):
            in_ch = base_filters * (2 ** (i-1))
            out_ch = base_filters * (2 ** i)
            self.encoder_blocks.append(ConvBlock(in_ch, out_ch))
        
        # Bottleneck
        bottleneck_in = base_filters * (2 ** (num_blocks-1))
        bottleneck_out = base_filters * (2 ** num_blocks)
        self.bottleneck = ConvBlock(bottleneck_in, bottleneck_out)
        
        # Decoder (Expanding path)
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        
        for i in range(num_blocks-1, -1, -1):
            # Upsampling block
            up_in = base_filters * (2 ** (i+1))
            up_out = base_filters * (2 ** i)
            self.upsample_blocks.append(
                nn.ConvTranspose2d(up_in, up_out, kernel_size=2, stride=2)
            )
            
            # Decoder block (after concatenation with skip connection)
            dec_in = up_out * 2  # Because we concatenate with skip connection
            dec_out = base_filters * (2 ** i)
            self.decoder_blocks.append(ConvBlock(dec_in, dec_out))
        
        # Final convolution
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for i in range(self.num_blocks):
            x = self.encoder_blocks[i](x)
            skip_connections.append(x)
            if i < self.num_blocks - 1:
                x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        for i in range(self.num_blocks):
            # Upsample
            x = self.upsample_blocks[i](x)
            
            # Get skip connection (in reverse order)
            skip = skip_connections[self.num_blocks - 1 - i]
            
            # Ensure spatial dimensions match
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            # Concatenate with skip connection
            x = torch.cat([x, skip], dim=1)
            
            # Conv block
            x = self.decoder_blocks[i](x)
        
        # Final output
        return torch.sigmoid(self.final_conv(x))

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.alpha = alpha
    
    def forward(self, predictions, targets):
        dice_loss = self.dice_loss(predictions, targets)
        bce_loss = self.bce_loss(predictions, targets)
        return self.alpha * dice_loss + (1 - self.alpha) * bce_loss

class RetinaSegmentationTorch:
    def __init__(self, data_paths):
        self.train_image_path = data_paths['train_image']
        self.train_mask_path = data_paths['train_mask']
        self.test_image_path = data_paths['test_image']
        self.test_mask_path = data_paths['test_mask']
    
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("Loading training data...")
        train_images = self.load_images(self.train_image_path, is_mask=False)
        train_masks = self.load_images(self.train_mask_path, is_mask=True)
        
        print("Loading test data...")
        test_images = self.load_images(self.test_image_path, is_mask=False)
        test_masks = self.load_images(self.test_mask_path, is_mask=True)
        
        # Normalize images
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        
        # Convert masks to binary
        train_masks = (train_masks > 0.5).astype(np.float32)
        test_masks = (test_masks > 0.5).astype(np.float32)
        
        # Split training data for validation
        train_images, val_images, train_masks, val_masks = train_test_split(
            train_images, train_masks, test_size=0.2, random_state=42
        )
        
        return (train_images, train_masks), (val_images, val_masks), (test_images, test_masks)
    
    def load_images(self, path, is_mask=False):
        """Load images from directory"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory not found: {path}")
            
        image_files = sorted([f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.tif', '.jpeg'))])
        if not image_files:
            raise FileNotFoundError(f"No images found in: {path}")
            
        images = []
        
        for img_file in tqdm(image_files, desc=f"Loading {os.path.basename(path)}"):
            img_path = os.path.join(path, img_file)
            if is_mask:
                # Load masks as grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            else:
                # Load images as RGB
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                else:
                    # If color image fails, try grayscale and convert to RGB
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = np.stack([img, img, img], axis=-1)  # Convert to 3-channel
            
            if img is not None:
                # Resize to 512x512 if needed
                if img.shape[:2] != (512, 512):
                    if len(img.shape) == 3:  # Color image
                        img = cv2.resize(img, (512, 512))
                    else:  # Grayscale image
                        img = cv2.resize(img, (512, 512))
                images.append(img)
            else:
                print(f"Warning: Could not load image {img_path}")
        
        return np.array(images).astype(np.float32)

class SimpleAugmentation:
    """Simple data augmentation using OpenCV"""
    def __init__(self):
        pass
    
    def __call__(self, image, mask):
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        # Random vertical flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (w, h))
            mask = cv2.warpAffine(mask, rotation_matrix, (w, h))
        
        # Random brightness for RGB images
        if np.random.random() > 0.5 and len(image.shape) == 3:
            brightness = np.random.uniform(0.9, 1.1)
            image = np.clip(image * brightness, 0, 255)
        
        return {'image': image, 'mask': mask}

def plot_training_progress(train_losses, val_dices, model_name):
    """Plot and save training progress - ONE graph per model"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training loss
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot validation dice
    ax2.plot(val_dices, label='Validation Dice', color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Coefficient')
    ax2.set_title(f'{model_name} - Validation Dice')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'training_screenshots/{model_name.replace(" ", "_").lower()}_training_progress.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training progress plot for {model_name}")

def calculate_dice_with_normalization(predictions, targets):
    """Calculate Dice coefficient with normalization (threshold at 0.5)"""
    predictions_bin = (predictions > 0.5).float()
    targets_bin = (targets > 0.5).float()
    
    intersection = (predictions_bin * targets_bin).sum()
    dice = (2. * intersection) / (predictions_bin.sum() + targets_bin.sum() + 1e-7)
    return dice.item()

def calculate_dice_without_normalization(predictions, targets):
    """Calculate Dice coefficient without normalization (using probabilities)"""
    predictions = predictions.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    
    intersection = (predictions * targets).sum()
    dice = (2. * intersection) / (predictions.sum() + targets.sum() + 1e-7)
    return dice.item()

def calculate_iou_with_normalization(predictions, targets):
    """Calculate IoU with normalization (threshold at 0.5)"""
    predictions_bin = (predictions > 0.5).float()
    targets_bin = (targets > 0.5).float()
    
    intersection = (predictions_bin * targets_bin).sum()
    union = predictions_bin.sum() + targets_bin.sum() - intersection
    iou = intersection / (union + 1e-7)
    return iou.item()

def calculate_iou_without_normalization(predictions, targets):
    """Calculate IoU without normalization (using probabilities)"""
    predictions = predictions.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    iou = intersection / (union + 1e-7)
    return iou.item()

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_name, config):
    """Train a single model"""
    best_val_dice = 0.0
    train_losses = []
    val_dices = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation phase
        model.eval()
        val_dice = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                dice = calculate_dice_with_normalization(outputs, masks)
                val_dice += dice
        
        avg_val_dice = val_dice / len(val_loader)
        val_dices.append(avg_val_dice)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Dice: {avg_val_dice:.4f}")
        
        # Save best model with config info
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            # Save model with config in filename for easy identification
            checkpoint = {
                'state_dict': model.state_dict(),
                'config': config,
                'val_dice': best_val_dice,
                'epoch': epoch,
                'train_losses': train_losses,
                'val_dices': val_dices
            }
            torch.save(checkpoint, f"best_{model_name.replace(' ', '_').lower()}.pth")
            print(f"Saved best model with Dice: {best_val_dice:.4f}")
    
    # Save ONE training progress graph at the end
    plot_training_progress(train_losses, val_dices, model_name)
    
    return train_losses, val_dices

def evaluate_model_comprehensive(model, test_loader, device):
    """Evaluate model on test set with both normalized and non-normalized metrics"""
    model.eval()
    dice_scores_norm = []
    dice_scores_non_norm = []
    iou_scores_norm = []
    iou_scores_non_norm = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            # Calculate metrics for each sample in batch
            for i in range(outputs.size(0)):
                dice_norm = calculate_dice_with_normalization(outputs[i], masks[i])
                dice_non_norm = calculate_dice_without_normalization(outputs[i], masks[i])
                iou_norm = calculate_iou_with_normalization(outputs[i], masks[i])
                iou_non_norm = calculate_iou_without_normalization(outputs[i], masks[i])
                
                dice_scores_norm.append(dice_norm)
                dice_scores_non_norm.append(dice_non_norm)
                iou_scores_norm.append(iou_norm)
                iou_scores_non_norm.append(iou_non_norm)
    
    return (np.mean(dice_scores_norm), np.mean(dice_scores_non_norm),
            np.mean(iou_scores_norm), np.mean(iou_scores_non_norm))

def test_model(model_path, test_loader, device, model_name):
    """Test a trained model and return predictions"""
    try:
        # Load checkpoint with config
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get config from checkpoint
        config = checkpoint['config']
        
        # Recreate model with exact same configuration
        model = MemoryEfficientUNet(
            num_blocks=config['blocks'],
            base_filters=config['filters']
        ).to(device)
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        print(f"Loaded model: {model_name} with blocks={config['blocks']}, filters={config['filters']}")
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                all_predictions.append(outputs.cpu())
                all_targets.append(masks.cpu())
        
        # Calculate comprehensive metrics
        dice_norm, dice_non_norm, iou_norm, iou_non_norm = evaluate_model_comprehensive(model, test_loader, device)
        
        print(f"{model_name} Results:")
        print(f"  Dice with normalization: {dice_norm:.4f}")
        print(f"  Dice without normalization: {dice_non_norm:.4f}")
        print(f"  IoU with normalization: {iou_norm:.4f}")
        print(f"  IoU without normalization: {iou_non_norm:.4f}")
        
        return (torch.cat(all_predictions, dim=0), torch.cat(all_targets, dim=0), 
                dice_norm, dice_non_norm, iou_norm, iou_non_norm)
        
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None, 0.0, 0.0, 0.0, 0.0

def visualize_test_predictions(test_images, test_masks, predictions, model_name, num_samples=4):
    """Visualize test predictions and save as screenshots"""
    fig, axes = plt.subplots(num_samples, 4, figsize=(15, 4*num_samples))
    
    # Handle different input types
    if isinstance(test_images, torch.Tensor):
        # Convert from CHW to HWC for visualization
        test_images_np = test_images.permute(0, 2, 3, 1).numpy()
    else:
        test_images_np = test_images
        
    if isinstance(test_masks, torch.Tensor):
        test_masks_np = test_masks.squeeze().numpy()
    else:
        test_masks_np = test_masks.squeeze()
    
    if predictions is not None:
        if isinstance(predictions, torch.Tensor):
            predictions_np = predictions.squeeze().numpy()
        else:
            predictions_np = predictions.squeeze()
    else:
        predictions_np = None
    
    for i in range(min(num_samples, len(test_images_np))):
        # Input image (RGB)
        axes[i, 0].imshow(test_images_np[i].astype(np.float32))
        axes[i, 0].set_title('Input Image (RGB)')
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(test_masks_np[i], cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        if predictions_np is not None:
            axes[i, 2].imshow(predictions_np[i], cmap='gray')
            axes[i, 2].set_title(f'{model_name} Prediction')
        else:
            axes[i, 2].text(0.5, 0.5, 'Prediction\nFailed', ha='center', va='center', 
                           transform=axes[i, 2].transAxes, fontsize=12)
        axes[i, 2].axis('off')
        
        # Overlay
        axes[i, 3].imshow(test_images_np[i].astype(np.float32))
        if predictions_np is not None:
            axes[i, 3].imshow(predictions_np[i], cmap='Reds', alpha=0.3)
            axes[i, 3].set_title('Overlay')
        else:
            axes[i, 3].text(0.5, 0.5, 'Overlay\nFailed', ha='center', va='center', 
                           transform=axes[i, 3].transAxes, fontsize=12)
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'test_screenshots/{model_name.replace(" ", "_").lower()}_predictions.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved test predictions for {model_name}")

def save_results_table(results, filename="results_summary.json"):
    """Save results in a structured format"""
    # Convert numpy types to Python native types for JSON serialization
    serializable_results = {}
    for model_name, metrics in results.items():
        serializable_results[model_name] = {
            'dice_with_norm': float(metrics['dice_norm']),
            'dice_without_norm': float(metrics['dice_non_norm']),
            'iou_with_norm': float(metrics['iou_norm']),
            'iou_without_norm': float(metrics['iou_non_norm']),
            'parameters': int(metrics['parameters'])
        }
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Also save as CSV for easy viewing
    try:
        import pandas as pd
        df_data = []
        for model_name, metrics in serializable_results.items():
            df_data.append({
                'Model': model_name,
                'Dice (with norm)': metrics['dice_with_norm'],
                'Dice (without norm)': metrics['dice_without_norm'],
                'IoU (with norm)': metrics['iou_with_norm'],
                'IoU (without norm)': metrics['iou_without_norm'],
                'Parameters': metrics['parameters']
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv('results_summary.csv', index=False)
        print("Results saved to results_summary.json and results_summary.csv")
    except ImportError:
        print("Pandas not available, saving only JSON results")
        print("Results saved to results_summary.json")

def main():
    # Data paths
    data_paths = {
        'train_image': 'Data/train/image',
        'train_mask': 'Data/train/mask',
        'test_image': 'Data/test/image',
        'test_mask': 'Data/test/mask'
    }
    
    # Check if data directories exist
    for path in data_paths.values():
        if not os.path.exists(path):
            print(f"Error: Data directory not found: {path}")
            print("Please check your data paths.")
            return
    
    # Initialize segmentation pipeline
    segmentation = RetinaSegmentationTorch(data_paths)
    
    try:
        # Load data
        (train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = \
            segmentation.load_and_preprocess_data()
        
        print(f"Training data - Images: {train_images.shape}, Masks: {train_masks.shape}")
        print(f"Validation data - Images: {val_images.shape}, Masks: {val_masks.shape}")
        print(f"Test data - Images: {test_images.shape}, Masks: {test_masks.shape}")
        
        # Check if images are RGB
        if len(train_images.shape) == 4:  # [N, H, W, C]
            print(f"Image shape: {train_images.shape}, appears to be RGB")
        else:
            print(f"Image shape: {train_images.shape}, converting to RGB")
            # Convert grayscale to RGB if needed
            if len(train_images.shape) == 3:
                train_images = np.stack([train_images] * 3, axis=-1)
                val_images = np.stack([val_images] * 3, axis=-1)
                test_images = np.stack([test_images] * 3, axis=-1)
        
        # Create datasets with augmentation for training
        augmentation = SimpleAugmentation()
        train_dataset = RetinaDataset(train_images, train_masks, 
                                    transform=augmentation, is_training=True)
        val_dataset = RetinaDataset(val_images, val_masks, is_training=False)
        test_dataset = RetinaDataset(test_images, test_masks, is_training=False)
        
        # Create data loaders with small batch sizes for memory efficiency
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
        
        # Model configurations - now with 3 input channels for RGB
        configurations = [
            {'name': 'U-Net 2 blocks', 'blocks': 2, 'filters': 16},
            {'name': 'U-Net 3 blocks', 'blocks': 3, 'filters': 16},
            {'name': 'U-Net 4 blocks', 'blocks': 4, 'filters': 16}
        ]
        
        results = {}
        
        for config in configurations:
            print(f"\n{'='*50}")
            print(f"Training {config['name']}")
            print(f"{'='*50}")
            
            # Create model with 3 input channels for RGB
            model = MemoryEfficientUNet(
                in_channels=3,  # RGB input
                num_blocks=config['blocks'],
                base_filters=config['filters']
            ).to(device)
            
            # Print model size and test forward pass
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {total_params:,}")
            
            # Test forward pass with dummy data (3 channels)
            try:
                dummy_input = torch.randn(2, 3, 512, 512).to(device)  # 3 channels for RGB
                dummy_output = model(dummy_input)
                print(f"Forward pass successful: input {dummy_input.shape} -> output {dummy_output.shape}")
            except Exception as e:
                print(f"Forward pass failed: {e}")
                continue
            
            # Loss and optimizer
            criterion = CombinedLoss(alpha=0.5)
            optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
            
            # Train model
            train_losses, val_dices = train_model(
                model, train_loader, val_loader, criterion, optimizer,
                num_epochs=30, device=device, model_name=config['name'], config=config
            )
            
            # Evaluate on test set with comprehensive metrics
            dice_norm, dice_non_norm, iou_norm, iou_non_norm = evaluate_model_comprehensive(model, test_loader, device)
            
            results[config['name']] = {
                'train_losses': train_losses,
                'val_dices': val_dices,
                'dice_norm': dice_norm,
                'dice_non_norm': dice_non_norm,
                'iou_norm': iou_norm,
                'iou_non_norm': iou_non_norm,
                'parameters': total_params
            }
            
            # Clear GPU memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Print final results in table format
        print(f"\n{'='*80}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"{'Model':<20} {'Dice (norm)':<12} {'Dice (non-norm)':<15} {'IoU (norm)':<12} {'IoU (non-norm)':<15} {'Params':<10}")
        print(f"{'-'*80}")
        
        for model_name, result in results.items():
            print(f"{model_name:<20} {result['dice_norm']:<12.4f} {result['dice_non_norm']:<15.4f} "
                  f"{result['iou_norm']:<12.4f} {result['iou_non_norm']:<15.4f} {result['parameters']:<10,}")
        
        # Save results to file
        save_results_table(results)
        
        # Test and visualize each model
        for config in configurations:
            model_name = config['name']
            model_path = f"best_{model_name.replace(' ', '_').lower()}.pth"
            
            if os.path.exists(model_path):
                predictions, targets, dice_norm, dice_non_norm, iou_norm, iou_non_norm = test_model(
                    model_path, test_loader, device, model_name
                )
                
                # Convert to numpy for visualization
                if predictions is not None:
                    visualize_test_predictions(test_images, test_masks, predictions, model_name)
        
        print(f"\nTraining completed! Check the following folders:")
        print(f"- training_screenshots/: Contains ONE training progress plot per model")
        print(f"- test_screenshots/: Contains test prediction visualizations")
        print(f"- results_summary.json & results_summary.csv: Contains comprehensive results")
                
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

# Simple test function as required
def test_model_simple(model_path, test_data_path):
    """Simple test function that loads model and predicts on test data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load checkpoint with config
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint['config']
        
        # Recreate model with exact same configuration
        model = MemoryEfficientUNet(
            in_channels=3,  # RGB input
            num_blocks=config['blocks'],
            base_filters=config['filters']
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        
        # Load test data as RGB

        data_paths = {
            'train_image': 'Data/train/image',
            'train_mask': 'Data/train/mask',
            'test_image': 'Data/test/image',
            'test_mask': 'Data/test/mask'
        }
        print("1")
        segmentation = RetinaSegmentationTorch(data_paths = data_paths)
        print("2")
        test_images = segmentation.load_images(test_data_path, is_mask=False) / 255.0
        
        predictions = []
        with torch.no_grad():
            for i in range(0, len(test_images), 2):  # Batch size 2
                batch_images = test_images[i:i+2]
                # Convert HWC to CHW for PyTorch
                batch_tensor = torch.from_numpy(batch_images).float().permute(0, 3, 1, 2).to(device)
                outputs = model(batch_tensor)
                predictions.append(outputs.cpu())
        
        return torch.cat(predictions, dim=0)
    except Exception as e:
        print(f"Error in test_model: {e}")
        return None

if __name__ == "__main__":
    main()