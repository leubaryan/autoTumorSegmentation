#!/usr/bin/env python3
"""
Setup script for Medical Image Segmentation Project
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def create_directories():
    """Create necessary project directories"""
    directories = [
        'data/train/images',
        'data/train/masks',
        'data/val/images',
        'data/val/masks',
        'data/test/images',
        'data/test/masks',
        'experiments',
        'results',
        'uploads',
        'logs',
        'checkpoints'
    ]
    
    print("\nCreating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠ CUDA not available - will use CPU")
            return False
    except ImportError:
        print("⚠ PyTorch not installed yet")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("\nInstalling Python dependencies...")
    
    # Upgrade pip
    run_command("python -m pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    if run_command("pip install -r requirements.txt", "Installing requirements"):
        print("✓ All dependencies installed successfully")
        return True
    else:
        print("✗ Failed to install some dependencies")
        return False

def create_sample_data():
    """Create sample data for testing"""
    print("\nCreating sample data...")
    
    try:
        import numpy as np
        from PIL import Image
        
        # Create sample images
        sample_dir = Path('data/sample')
        sample_dir.mkdir(exist_ok=True)
        
        # Create a simple test image
        img_array = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(sample_dir / 'sample_image.png')
        
        # Create a simple test mask
        mask_array = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
        mask = Image.fromarray(mask_array)
        mask.save(sample_dir / 'sample_mask.png')
        
        print("✓ Sample data created in data/sample/")
        return True
    except Exception as e:
        print(f"✗ Failed to create sample data: {e}")
        return False

def test_installation():
    """Test the installation"""
    print("\nTesting installation...")
    
    try:
        # Test imports
        import torch
        import torchvision
        import numpy as np
        import matplotlib.pyplot as plt
        import nibabel as nib
        import cv2
        from PIL import Image
        
        print("✓ All core libraries imported successfully")
        
        # Test model creation
        sys.path.append('models')
        from unet import UNet
        
        model = UNet(n_channels=1, n_classes=1)
        test_input = torch.randn(1, 1, 256, 256)
        output = model(test_input)
        
        print(f"✓ Model test successful - Input: {test_input.shape}, Output: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Installation test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Prepare your dataset:")
    print("   - Place training images in data/train/images/")
    print("   - Place training masks in data/train/masks/")
    print("   - Place validation images in data/val/images/")
    print("   - Place validation masks in data/val/masks/")
    print("\n2. Configure training:")
    print("   - Edit config/training_config.yaml")
    print("   - Adjust model parameters and training settings")
    print("\n3. Start training:")
    print("   python training/train.py --config config/training_config.yaml")
    print("\n4. Explore data:")
    print("   jupyter notebook notebooks/01_data_exploration.ipynb")
    print("\n5. Run web interface:")
    print("   python web_app/app.py")
    print("\n6. For BraTS dataset:")
    print("   - Download from https://www.med.upenn.edu/cbica/brats2021/")
    print("   - Extract to data/brats/")
    print("   - Update config to use 'brats' data type")
    print("\nUseful commands:")
    print("- Check GPU: python -c 'import torch; print(torch.cuda.is_available())'")
    print("- Test model: python models/unet.py")
    print("- Health check: python web_app/app.py (then visit /health)")

def main():
    """Main setup function"""
    print("Medical Image Segmentation Project Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check CUDA
    check_cuda()
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("\n⚠ Some dependencies failed to install. You may need to install them manually.")
        print("Try: pip install torch torchvision torchaudio")
    
    # Create sample data
    create_sample_data()
    
    # Test installation
    if test_installation():
        print("\n✓ Installation test passed!")
    else:
        print("\n⚠ Installation test failed. Check the errors above.")
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main() 