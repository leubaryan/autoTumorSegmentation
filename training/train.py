import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import wandb
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import UNet
from utils.dataset import MedicalImageDataset, BraTSDataset, create_data_loaders
from utils.losses import DiceLoss, CombinedLoss
from utils.metrics import calculate_metrics


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize model
        self.model = self.create_model()
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self.create_data_loaders()
        
        # Initialize optimizer and scheduler
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        
        # Initialize loss function
        self.criterion = self.create_loss_function()
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = 0.0
        
    def setup_logging(self):
        """Setup logging with TensorBoard and WandB"""
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(self.config['training']['save_dir'], f"exp_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, 'tensorboard'))
        
        # WandB (optional)
        if self.config['training'].get('use_wandb', False):
            wandb.init(
                project=self.config['training'].get('wandb_project', 'medical-segmentation'),
                config=self.config,
                name=f"exp_{timestamp}"
            )
    
    def create_model(self):
        """Create and initialize the model"""
        model_config = self.config['model']
        
        if model_config['type'] == 'unet':
            model = UNet(
                n_channels=model_config.get('n_channels', 1),
                n_classes=model_config.get('n_classes', 1),
                bilinear=model_config.get('bilinear', False),
                use_attention=model_config.get('use_attention', False)
            )
        else:
            raise ValueError(f"Unknown model type: {model_config['type']}")
        
        model = model.to(self.device)
        
        # Load pretrained weights if specified
        if 'pretrained_path' in model_config:
            checkpoint = torch.load(model_config['pretrained_path'], map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded pretrained weights from {model_config['pretrained_path']}")
        
        return model
    
    def create_data_loaders(self):
        """Create training and validation data loaders"""
        data_config = self.config['data']
        
        if data_config['type'] == 'brats':
            train_dataset = BraTSDataset(
                data_dir=data_config['train_dir'],
                image_size=data_config['image_size'],
                mode='train'
            )
            val_dataset = BraTSDataset(
                data_dir=data_config['val_dir'],
                image_size=data_config['image_size'],
                mode='val'
            )
        else:
            train_dataset = MedicalImageDataset(
                image_dir=data_config['train_image_dir'],
                mask_dir=data_config['train_mask_dir'],
                image_size=data_config['image_size'],
                mode='train'
            )
            val_dataset = MedicalImageDataset(
                image_dir=data_config['val_image_dir'],
                mask_dir=data_config['val_mask_dir'],
                image_size=data_config['image_size'],
                mode='val'
            )
        
        train_loader, val_loader = create_data_loaders(
            train_dataset, val_dataset,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training'].get('num_workers', 4)
        )
        
        return train_loader, val_loader
    
    def create_optimizer(self):
        """Create optimizer"""
        optimizer_config = self.config['training']['optimizer']
        
        if optimizer_config['type'] == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0.0)
            )
        elif optimizer_config['type'] == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0.0)
            )
        elif optimizer_config['type'] == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_config['type']}")
        
        return optimizer
    
    def create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_config = self.config['training'].get('scheduler', {})
        
        if not scheduler_config:
            return None
        
        if scheduler_config['type'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_config['type'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config['T_max']
            )
        elif scheduler_config['type'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config.get('factor', 0.1),
                patience=scheduler_config.get('patience', 10)
            )
        else:
            return None
        
        return scheduler
    
    def create_loss_function(self):
        """Create loss function"""
        loss_config = self.config['training']['loss']
        
        if loss_config['type'] == 'dice':
            return DiceLoss()
        elif loss_config['type'] == 'combined':
            return CombinedLoss(
                dice_weight=loss_config.get('dice_weight', 0.5),
                bce_weight=loss_config.get('bce_weight', 0.5)
            )
        elif loss_config['type'] == 'bce':
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_config['type']}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_metrics = {'dice': 0.0, 'iou': 0.0}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                predictions = torch.sigmoid(outputs) > 0.5
                metrics = calculate_metrics(predictions, masks)
            
            # Update running totals
            total_loss += loss.item()
            total_metrics['dice'] += metrics['dice']
            total_metrics['iou'] += metrics['iou']
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Dice': f"{metrics['dice']:.4f}",
                'IoU': f"{metrics['iou']:.4f}"
            })
        
        # Calculate averages
        avg_loss = total_loss / len(self.train_loader)
        avg_metrics = {k: v / len(self.train_loader) for k, v in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_metrics = {'dice': 0.0, 'iou': 0.0}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Calculate metrics
                predictions = torch.sigmoid(outputs) > 0.5
                metrics = calculate_metrics(predictions, masks)
                
                # Update running totals
                total_loss += loss.item()
                total_metrics['dice'] += metrics['dice']
                total_metrics['iou'] += metrics['iou']
        
        # Calculate averages
        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = {k: v / len(self.val_loader) for k, v in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_score': self.best_val_score,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.exp_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.exp_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation score: {self.best_val_score:.4f}")
    
    def log_metrics(self, train_loss, train_metrics, val_loss, val_metrics):
        """Log metrics to TensorBoard and WandB"""
        # TensorBoard
        self.writer.add_scalar('Loss/Train', train_loss, self.current_epoch)
        self.writer.add_scalar('Loss/Val', val_loss, self.current_epoch)
        self.writer.add_scalar('Dice/Train', train_metrics['dice'], self.current_epoch)
        self.writer.add_scalar('Dice/Val', val_metrics['dice'], self.current_epoch)
        self.writer.add_scalar('IoU/Train', train_metrics['iou'], self.current_epoch)
        self.writer.add_scalar('IoU/Val', val_metrics['iou'], self.current_epoch)
        self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        
        # WandB
        if self.config['training'].get('use_wandb', False):
            wandb.log({
                'epoch': self.current_epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_dice': train_metrics['dice'],
                'val_dice': val_metrics['dice'],
                'train_iou': train_metrics['iou'],
                'val_iou': val_metrics['iou'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
    
    def train(self):
        """Main training loop"""
        print(f"Starting training on device: {self.device}")
        print(f"Training for {self.config['training']['epochs']} epochs")
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['dice'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.log_metrics(train_loss, train_metrics, val_loss, val_metrics)
            
            # Save checkpoint
            is_best = val_metrics['dice'] > self.best_val_score
            if is_best:
                self.best_val_score = val_metrics['dice']
            
            self.save_checkpoint(is_best)
            
            # Print epoch summary
            print(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")
            print(f"Train - Loss: {train_loss:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
            print(f"Best Val Dice: {self.best_val_score:.4f}")
            print("-" * 50)
        
        print(f"Training completed! Best validation Dice: {self.best_val_score:.4f}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train medical image segmentation model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main() 