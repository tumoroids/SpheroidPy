"""
Training utilities for HRNet-W30 segmentation model.

This module provides functions to:
- Load pretrained weights for fine-tuning
- Create training datasets
- Train the model from scratch or continue training
- Evaluate model performance
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import numpy as np
import cv2

logger = logging.getLogger("SpheroidPy.utils.hrnet_training")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    Dataset = None
    DataLoader = None

from .hrnet_utils import HRNetSeg, preprocess_image_torch, BN_MOMENTUM


class SegmentationDataset(Dataset):
    """PyTorch Dataset for segmentation training.
    
    Expects images and masks in the same directory structure:
    - images/: Input images (any format readable by cv2)
    - masks/: Binary masks (0=background, 1=foreground)
    
    Args:
        image_dir: Directory containing input images
        mask_dir: Directory containing ground truth masks
        transform: Optional transform function (applied to both image and mask)
    """
    
    def __init__(self, image_dir: Path, mask_dir: Path, transform=None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training.")
        
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        
        # Get all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend(list(self.image_dir.glob(f'*{ext}')))
            self.image_files.extend(list(self.image_dir.glob(f'*{ext.upper()}')))
        
        self.image_files = sorted(self.image_files)
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        logger.info(f"Found {len(self.image_files)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        from .image_io import imread
        image_path = self.image_files[idx]
        image = imread(str(image_path), cv2.IMREAD_ANYDEPTH)
        
        if image is None:
            raise IOError(f"Failed to load image: {image_path}")
        
        # Load corresponding mask
        mask_path = self.mask_dir / image_path.name
        if not mask_path.exists():
            # Try with different extensions
            mask_path = None
            for ext in ['.png', '.tif', '.tiff', '.jpg']:
                candidate = self.mask_dir / (image_path.stem + ext)
                if candidate.exists():
                    mask_path = candidate
                    break
            
            if mask_path is None:
                raise FileNotFoundError(f"Mask not found for {image_path.name}")
        
        mask = imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Failed to load mask: {mask_path}")
        
        # Convert mask to binary (0 or 1)
        mask = (mask > 127).astype(np.uint8)
        
        # Apply transforms if provided
        if self.transform:
            image, mask = self.transform(image, mask)
        
        # Convert to torch tensors
        # Image: preprocess (grayscale→RGB, normalize to [0,1])
        image_t = preprocess_image_torch(image, device="cpu")
        image_t = image_t.squeeze(0)  # Remove batch dimension for dataset
        
        # Mask: convert to long tensor with class indices (0=background, 1=foreground)
        mask_t = torch.from_numpy(mask).long()
        
        return image_t, mask_t


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation.
    
    Dice Loss = 1 - (2 * |X ∩ Y|) / (|X| + |Y|)
    """
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # predictions: [B, C, H, W] (logits)
        # targets: [B, H, W] (class indices)
        
        # Apply softmax to get probabilities
        probs = F.softmax(predictions, dim=1)
        
        # Get foreground probability (class 1)
        pred_foreground = probs[:, 1, :, :]  # [B, H, W]
        
        # Convert targets to one-hot: [B, H, W] -> [B, H, W] with 1 for foreground
        target_foreground = (targets == 1).float()  # [B, H, W]
        
        # Flatten
        pred_flat = pred_foreground.contiguous().view(-1)
        target_flat = target_foreground.contiguous().view(-1)
        
        # Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined Cross-Entropy and Dice Loss.
    
    Useful for segmentation tasks as it combines pixel-wise and region-based losses.
    """
    
    def __init__(self, ce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(smooth=smooth)
    
    def forward(self, predictions, targets):
        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        return self.ce_weight * ce + self.dice_weight * dice


def load_model_for_training(
    weights_path: Optional[Path | str] = None,
    num_classes: int = 2,
    freeze_backbone: bool = False,
    device: str = "cpu"
) -> Tuple[HRNetSeg, Dict]:
    """Load HRNet model for training (not inference).
    
    Args:
        weights_path: Path to pretrained weights (state_dict format)
        num_classes: Number of segmentation classes (default: 2 for binary)
        freeze_backbone: If True, freeze backbone weights (only train head)
        device: Target device ('cpu' or 'cuda')
        
    Returns:
        Tuple of (model, info_dict) where info_dict contains loading information
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training.")
    
    # Create model
    model = HRNetSeg(num_classes=num_classes, backbone_name="hrnet_w30")
    info = {
        'loaded_weights': False,
        'frozen_layers': [],
        'trainable_layers': []
    }
    
    # Load pretrained weights if provided
    if weights_path is not None:
        weights_path = Path(weights_path)
        if weights_path.exists():
            logger.info(f"Loading pretrained weights from: {weights_path}")
            state_dict = torch.load(str(weights_path), map_location=device)
            
            if isinstance(state_dict, dict):
                # Try to load with strict=False to allow partial matching
                try:
                    model.load_state_dict(state_dict, strict=True)
                    logger.info("Pretrained weights loaded successfully (strict=True)")
                except RuntimeError as e:
                    logger.warning(f"Strict loading failed: {e}. Trying strict=False...")
                    model.load_state_dict(state_dict, strict=False)
                    logger.info("Pretrained weights loaded successfully (strict=False)")
                info['loaded_weights'] = True
            else:
                logger.warning(f"Expected state_dict, got {type(state_dict)}. Starting from scratch.")
    
    # Freeze backbone if requested
    if freeze_backbone:
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False
            info['frozen_layers'].append(name)
        logger.info(f"Frozen {len(info['frozen_layers'])} backbone parameters")
    
    # Track trainable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            info['trainable_layers'].append(name)
    
    model = model.to(device)
    model.train()  # Set to training mode
    
    # Log parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
    
    return model, info


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str = "cpu",
    accumulation_steps: int = 1
) -> Dict[str, float]:
    """Train for one epoch.
    
    Returns:
        Dictionary with training metrics (loss, accuracy, etc.)
    """
    model.train()
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)  # [B, num_classes, H, W]
        
        # Calculate loss
        loss = criterion(outputs, masks)
        
        # Gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Metrics
        running_loss += loss.item() * accumulation_steps
        
        # Pixel accuracy
        pred_classes = torch.argmax(outputs, dim=1)  # [B, H, W]
        correct_pixels += (pred_classes == masks).sum().item()
        total_pixels += masks.numel()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu"
) -> Dict[str, float]:
    """Validate model.
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            
            pred_classes = torch.argmax(outputs, dim=1)
            correct_pixels += (pred_classes == masks).sum().item()
            total_pixels += masks.numel()
    
    val_loss = running_loss / len(dataloader)
    val_acc = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    
    return {
        'loss': val_loss,
        'accuracy': val_acc
    }


def train_hrnet(
    train_image_dir: Path | str,
    train_mask_dir: Path | str,
    val_image_dir: Optional[Path | str] = None,
    val_mask_dir: Optional[Path | str] = None,
    pretrained_weights: Optional[Path | str] = None,
    output_dir: Path | str = "./checkpoints",
    num_epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    freeze_backbone: bool = False,
    num_classes: int = 2,
    device: str = "cuda",
    save_best: bool = True,
    save_every: int = 10,
    loss_type: str = "combined",  # "ce", "dice", "combined"
    scheduler_type: str = "step",  # "step", "cosine", None
    accumulation_steps: int = 1
) -> HRNetSeg:
    """Main training function for HRNet-W30.
    
    Args:
        train_image_dir: Directory with training images
        train_mask_dir: Directory with training masks
        val_image_dir: Optional validation image directory
        val_mask_dir: Optional validation mask directory
        pretrained_weights: Path to pretrained weights (state_dict) for fine-tuning
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        freeze_backbone: If True, only train segmentation head
        num_classes: Number of classes (default: 2 for binary)
        device: Training device ('cpu' or 'cuda')
        save_best: Save best model based on validation loss
        save_every: Save checkpoint every N epochs
        loss_type: Loss function ('ce', 'dice', 'combined')
        scheduler_type: Learning rate scheduler ('step', 'cosine', None)
        accumulation_steps: Gradient accumulation steps (for larger effective batch size)
        
    Returns:
        Trained model
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training.")
    
    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    logger.info(f"Training on device: {device}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    train_dataset = SegmentationDataset(train_image_dir, train_mask_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    val_loader = None
    if val_image_dir and val_mask_dir:
        val_dataset = SegmentationDataset(val_image_dir, val_mask_dir)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if device == "cuda" else False
        )
    
    # Load model
    model, load_info = load_model_for_training(
        weights_path=pretrained_weights,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        device=device
    )
    logger.info(f"Model loaded. Trainable parameters: {len(load_info['trainable_layers'])}")
    
    # Loss function
    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    elif loss_type == "dice":
        criterion = DiceLoss()
    elif loss_type == "combined":
        criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = None
    if scheduler_type == "step":
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, accumulation_steps
        )
        
        # Validate
        val_metrics = None
        if val_loader:
            val_metrics = validate(model, val_loader, criterion, device)
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
        else:
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}"
            )
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step()
            logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if save_every > 0 and (epoch + 1) % save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if save_best and val_metrics and val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_path = output_dir / "best_model.pth"
            torch.save(model.state_dict(), best_path)
            logger.info(f"Best model saved: {best_path} (val_loss: {best_val_loss:.4f})")
        
        # Record history
        history_entry = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy']
        }
        if val_metrics:
            history_entry.update({
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy']
            })
        training_history.append(history_entry)
    
    # Save final model
    final_path = output_dir / "final_model.pth"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Final model saved: {final_path}")
    
    # Save training history
    import json
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Training history saved: {history_path}")
    
    return model

