import os
import io
import base64
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import cv2
import nibabel as nib
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import UNet
from utils.metrics import calculate_metrics, print_metrics_summary

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global variables
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'nii', 'nii.gz', 'dcm'}
MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model(model_path):
    """Load the trained model"""
    global MODEL
    
    if MODEL is None:
        # Load configuration from checkpoint
        checkpoint = torch.load(model_path, map_location=DEVICE)
        config = checkpoint.get('config', {})
        
        # Create model
        model_config = config.get('model', {})
        MODEL = UNet(
            n_channels=model_config.get('n_channels', 1),
            n_classes=model_config.get('n_classes', 1),
            bilinear=model_config.get('bilinear', False),
            use_attention=model_config.get('use_attention', False)
        )
        
        # Load weights
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL.to(DEVICE)
        MODEL.eval()
        
        print(f"Model loaded successfully from {model_path}")
    
    return MODEL

def preprocess_image(image_path):
    """Preprocess uploaded image for model inference"""
    # Determine file type and load accordingly
    if image_path.endswith(('.nii', '.nii.gz')):
        # Load NIfTI file
        img = nib.load(image_path)
        data = img.get_fdata()
        
        # Normalize to [0, 1]
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        # Convert to PIL Image for resizing
        data = (data * 255).astype(np.uint8)
        image = Image.fromarray(data)
        
    elif image_path.endswith('.dcm'):
        # Load DICOM file
        import pydicom
        ds = pydicom.dcmread(image_path)
        data = ds.pixel_array
        
        # Normalize
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        data = (data * 255).astype(np.uint8)
        image = Image.fromarray(data)
        
    else:
        # Load regular image
        image = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Resize to model input size
    image = image.resize((256, 256), Image.LANCZOS)
    
    # Convert to tensor
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    
    return image_tensor, image_array

def postprocess_prediction(prediction, threshold=0.5):
    """Post-process model prediction"""
    # Apply sigmoid and threshold
    prediction = torch.sigmoid(prediction)
    binary_pred = (prediction > threshold).float()
    
    # Convert to numpy
    pred_array = binary_pred.squeeze().cpu().numpy()
    
    return pred_array

def create_visualization(original, prediction, ground_truth=None):
    """Create visualization of results"""
    fig, axes = plt.subplots(1, 3 if ground_truth is not None else 2, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(prediction, cmap='hot', alpha=0.7)
    axes[1].imshow(original, cmap='gray', alpha=0.3)
    axes[1].set_title('Segmentation Result')
    axes[1].axis('off')
    
    if ground_truth is not None:
        # Ground truth
        axes[2].imshow(ground_truth, cmap='hot', alpha=0.7)
        axes[2].imshow(original, cmap='gray', alpha=0.3)
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save to bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

def encode_image_to_base64(image_array):
    """Encode numpy array to base64 string"""
    # Normalize to 0-255 range
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    image = Image.fromarray(image_array)
    
    # Save to bytes
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Encode to base64
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and segmentation"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load model (you'll need to specify the path to your trained model)
        model_path = request.form.get('model_path', 'experiments/best_model.pth')
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found. Please train a model first.'}), 400
        
        model = load_model(model_path)
        
        # Preprocess image
        image_tensor, original_image = preprocess_image(filepath)
        image_tensor = image_tensor.to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            prediction = model(image_tensor)
        
        # Post-process prediction
        pred_array = postprocess_prediction(prediction)
        
        # Create visualization
        viz_buffer = create_visualization(original_image, pred_array)
        
        # Encode images to base64
        original_b64 = encode_image_to_base64(original_image)
        prediction_b64 = encode_image_to_base64(pred_array)
        
        # Calculate metrics if ground truth is provided
        metrics = None
        if 'ground_truth' in request.files:
            gt_file = request.files['ground_truth']
            if gt_file.filename != '':
                gt_filename = secure_filename(gt_file.filename)
                gt_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gt_filename)
                gt_file.save(gt_filepath)
                
                # Load and preprocess ground truth
                gt_tensor, gt_image = preprocess_image(gt_filepath)
                gt_array = (gt_image > 0.5).astype(np.float32)
                
                # Calculate metrics
                pred_tensor = torch.from_numpy(pred_array).unsqueeze(0).unsqueeze(0)
                gt_tensor = torch.from_numpy(gt_array).unsqueeze(0).unsqueeze(0)
                
                metrics = calculate_metrics(pred_tensor, gt_tensor)
                
                # Create visualization with ground truth
                viz_buffer = create_visualization(original_image, pred_array, gt_array)
        
        # Save results
        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        
        # Save prediction as image
        pred_image = Image.fromarray((pred_array * 255).astype(np.uint8))
        pred_image.save(result_path)
        
        # Prepare response
        response = {
            'success': True,
            'original_image': original_b64,
            'prediction_image': prediction_b64,
            'result_path': result_path,
            'metrics': metrics
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download result file"""
    try:
        return send_file(
            os.path.join(app.config['RESULTS_FOLDER'], filename),
            as_attachment=True
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/models')
def list_models():
    """List available trained models"""
    models_dir = 'experiments'
    models = []
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.pth'):
                models.append({
                    'name': file,
                    'path': os.path.join(models_dir, file),
                    'size': os.path.getsize(os.path.join(models_dir, file))
                })
    
    return jsonify({'models': models})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(DEVICE),
        'model_loaded': MODEL is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 