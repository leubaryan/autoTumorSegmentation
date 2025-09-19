import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from PIL import Image
import cv2
from typing import Tuple, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MedicalImageDataset(Dataset):
    """Dataset class for medical image segmentation"""
    
    def __init__(self, 
                 image_dir: str,
                 mask_dir: str,
                 transform=None,
                 image_size: Tuple[int, int] = (256, 256),
                 mode: str = 'train'):
        """
        Args:
            image_dir: Directory with image files
            mask_dir: Directory with mask files
            transform: Optional transform to be applied
            image_size: Target image size (height, width)
            mode: 'train', 'val', or 'test'
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        
        # Get file paths
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                  if f.endswith(('.nii.gz', '.nii', '.png', '.jpg', '.jpeg'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) 
                                 if f.endswith(('.nii.gz', '.nii', '.png', '.jpg', '.jpeg'))])
        
        # Set up transforms
        if self.transform is None:
            self.transform = self._get_default_transforms()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()
        
        # Ensure correct dimensions
        if len(image.shape) == 2:
            image = image.unsqueeze(0)  # Add channel dimension
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # Add channel dimension
        
        return {
            'image': image,
            'mask': mask,
            'filename': self.image_files[idx]
        }
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load image from various formats"""
        if path.endswith(('.nii.gz', '.nii')):
            # Load NIfTI file
            img = nib.load(path)
            data = img.get_fdata()
            # Normalize to [0, 1]
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            return data.astype(np.float32)
        else:
            # Load regular image
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image: {path}")
            img = img.astype(np.float32) / 255.0
            return img
    
    def _load_mask(self, path: str) -> np.ndarray:
        """Load mask from various formats"""
        if path.endswith(('.nii.gz', '.nii')):
            # Load NIfTI file
            mask = nib.load(path)
            data = mask.get_fdata()
            # Binarize mask
            data = (data > 0).astype(np.float32)
            return data
        else:
            # Load regular image
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {path}")
            mask = (mask > 127).astype(np.float32)
            return mask
    
    def _get_default_transforms(self):
        """Get default transforms based on mode"""
        if self.mode == 'train':
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomGamma(p=0.5),
                ], p=0.3),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])


class BraTSDataset(Dataset):
    """Specific dataset for BraTS brain tumor segmentation"""
    
    def __init__(self, 
                 data_dir: str,
                 transform=None,
                 image_size: Tuple[int, int] = (256, 256),
                 mode: str = 'train'):
        """
        Args:
            data_dir: Directory containing BraTS data
            transform: Optional transform to be applied
            image_size: Target image size
            mode: 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        
        # Get patient directories
        self.patients = [d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))]
        
        # Set up transforms
        if self.transform is None:
            self.transform = self._get_default_transforms()
    
    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        patient_dir = os.path.join(self.data_dir, patient_id)
        
        # Load different MRI modalities
        t1_path = os.path.join(patient_dir, f"{patient_id}_t1.nii.gz")
        t1ce_path = os.path.join(patient_dir, f"{patient_id}_t1ce.nii.gz")
        t2_path = os.path.join(patient_dir, f"{patient_id}_t2.nii.gz")
        flair_path = os.path.join(patient_dir, f"{patient_id}_flair.nii.gz")
        seg_path = os.path.join(patient_dir, f"{patient_id}_seg.nii.gz")
        
        # Load images
        t1 = self._load_nifti(t1_path)
        t1ce = self._load_nifti(t1ce_path)
        t2 = self._load_nifti(t2_path)
        flair = self._load_nifti(flair_path)
        seg = self._load_nifti(seg_path)
        
        # Stack modalities
        image = np.stack([t1, t1ce, t2, flair], axis=0)  # 4 channels
        
        # Process segmentation mask
        mask = self._process_segmentation(seg)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return {
            'image': image,
            'mask': mask,
            'patient_id': patient_id
        }
    
    def _load_nifti(self, path: str) -> np.ndarray:
        """Load NIfTI file and normalize"""
        img = nib.load(path)
        data = img.get_fdata()
        # Normalize to [0, 1]
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        return data.astype(np.float32)
    
    def _process_segmentation(self, seg: np.ndarray) -> np.ndarray:
        """Process BraTS segmentation labels"""
        # BraTS labels: 1=necrotic, 2=edema, 4=enhancing tumor
        # Combine into binary mask (tumor vs background)
        mask = (seg > 0).astype(np.float32)
        return mask
    
    def _get_default_transforms(self):
        """Get default transforms for BraTS data"""
        if self.mode == 'train':
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomGamma(p=0.5),
                ], p=0.3),
                A.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5]),
                ToTensorV2(),
            ])


def create_data_loaders(train_dataset, val_dataset, batch_size=8, num_workers=4):
    """Create data loaders for training and validation"""
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing MedicalImageDataset...")
    # This would require actual data files to test
    print("Dataset classes created successfully!") 