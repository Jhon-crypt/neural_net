# MNIST Handwritten Digit Classification

A comprehensive neural network implementation for classifying handwritten digits using the MNIST dataset and custom images with TensorFlow/Keras.

## 🎯 Overview

This project implements a deep neural network to classify handwritten digits (0-9) from both the famous MNIST dataset and custom user images. The enhanced classifier supports real-world image processing and achieves high accuracy with a flexible, production-ready architecture.

## ✨ Features

### Core Functionality
- **MNIST Training**: Automatic loading and training on the classic MNIST dataset
- **Custom Image Classification**: Process your own handwritten digit images
- **Batch Processing**: Classify entire directories of images at once
- **Flexible Architecture**: Configurable hidden layers and dropout rates
- **Advanced Training**: Early stopping and learning rate reduction callbacks

### Image Processing
- **Smart Preprocessing**: Automatic resizing, normalization, and inversion detection
- **Multiple Formats**: Support for PNG, JPG, JPEG, BMP, TIFF
- **Robust Handling**: Error handling for various image formats and qualities

### Visualization & Analysis
- **Training Metrics**: Comprehensive accuracy and loss plots
- **Confusion Matrix**: Detailed classification performance analysis
- **Prediction Visualization**: Visual comparison of predictions vs actual digits
- **Custom Image Visualization**: See how your images are classified
- **Confidence Scores**: Get prediction confidence for each classification

### Model Management
- **Modern Format**: Uses latest Keras .keras format (no deprecation warnings)
- **Model Persistence**: Save and load trained models
- **Organized Structure**: Clean project organization with separate directories

## 📁 Project Structure

```
neural_net/
├── src/                          # Source code
│   └── mnist_classifier.py      # Main classifier implementation
├── notebooks/                    # Interactive Jupyter notebooks
│   ├── training_demo.ipynb      # Step-by-step training demonstration
│   └── analysis_demo.ipynb      # Comprehensive model analysis
├── examples/                     # Usage examples
│   ├── example.py               # Basic MNIST training example
│   └── classify_custom_images.py # Custom image classification examples
├── tests/                        # Testing suite
│   └── test_basic.py            # Automated system tests
├── models/                       # Saved trained models
├── assets/                       # Generated outputs and sample images
│   ├── sample_digits/           # Sample digit images for testing
│   ├── training_history.png     # Training metrics visualization
│   ├── confusion_matrix.png     # Classification performance
│   ├── sample_predictions.png   # MNIST test predictions
│   └── custom_predictions.png   # Custom image predictions
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                    # This comprehensive guide
```

## 📓 Interactive Notebooks

For the best learning experience, check out our Jupyter notebooks:

### 🎯 Training Demo (`notebooks/training_demo.ipynb`)
Interactive step-by-step training demonstration:
- Dataset exploration and visualization
- Model architecture explanation
- Live training progress monitoring
- Performance evaluation and model saving

### 📊 Analysis Demo (`notebooks/analysis_demo.ipynb`)
Comprehensive model analysis and evaluation:
- Detailed performance metrics
- Confusion matrix analysis
- Error analysis with misclassified examples
- Confidence score distributions
- Model interpretation insights

**To run notebooks:**
```bash
# Install Jupyter if not already installed
pip install jupyter

# Start Jupyter server
jupyter notebook

# Navigate to notebooks/ folder and open desired notebook
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or navigate to project directory
cd neural_net

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Run basic training example
python examples/example.py

# Or run full training pipeline
python src/mnist_classifier.py
```

### 3. Classify Your Own Images

```bash
# Run custom image classification examples
python examples/classify_custom_images.py
```

## 💻 Usage Examples

### Basic MNIST Training

```python
from src.mnist_classifier import MNISTClassifier

# Create and train classifier
classifier = MNISTClassifier()
classifier.load_and_preprocess_data()
classifier.build_model(hidden_layers=[128, 64], dropout_rate=0.2)
classifier.train_model(epochs=20, batch_size=128)

# Evaluate and visualize
test_accuracy, y_pred, y_true = classifier.evaluate_model()
classifier.plot_training_history()
classifier.plot_confusion_matrix(y_true, y_pred)
classifier.visualize_predictions(num_samples=20)

# Save the model
classifier.save_model("models/my_mnist_model.keras")
```

### Custom Image Classification

```python
from src.mnist_classifier import MNISTClassifier

# Load pre-trained model
classifier = MNISTClassifier()
classifier.load_model("models/mnist_model.keras")

# Classify single image
digit, confidence = classifier.classify_image("path/to/your/digit.png")
print(f"Predicted digit: {digit} (confidence: {confidence:.3f})")

# Classify all images in directory
results = classifier.classify_images_in_directory("path/to/image/directory")

# Visualize predictions
image_files = ["image1.png", "image2.png", "image3.png"]
classifier.visualize_custom_predictions(image_files)
```

### Advanced Configuration

```python
# Custom architecture
classifier.build_model(hidden_layers=[256, 128, 64], dropout_rate=0.3)

# Custom training parameters
classifier.train_model(epochs=30, batch_size=64, validation_split=0.15)

# Process specific image formats
results = classifier.classify_images_in_directory(
    "my_images/", 
    image_extensions=['*.png', '*.jpg']
)
```

## 🧠 Model Architecture

The default neural network architecture includes:

- **Input Layer**: 784 neurons (28×28 flattened pixels)
- **Hidden Layer 1**: 128 neurons with ReLU activation + Dropout (0.2)
- **Hidden Layer 2**: 64 neurons with ReLU activation + Dropout (0.2)  
- **Output Layer**: 10 neurons with Softmax activation (for 10 digit classes)

### Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Callbacks**: 
  - Early Stopping (patience=5)
  - Learning Rate Reduction (factor=0.5, patience=3)
- **Default Parameters**:
  - Epochs: 20
  - Batch Size: 128
  - Validation Split: 10%

## 📊 Expected Performance

### MNIST Dataset
- **Training Accuracy**: ~99%
- **Test Accuracy**: ~97-98%
- **Training Time**: 2-5 minutes (CPU), 1-2 minutes (GPU)

### Custom Images
- **Accuracy**: Depends on image quality and similarity to MNIST style
- **Best Results**: Clean, centered, single digits on light backgrounds
- **Processing Speed**: ~10-50 images per second

## 🧪 Testing Guide

### Automated Testing

```bash
# Run comprehensive system tests
python tests/test_basic.py
```

**Expected Results**: All 4/4 tests should pass:
- ✅ Import verification
- ✅ MNIST data loading
- ✅ Model building
- ✅ GPU availability check

### Manual Testing Levels

#### 1. Quick Demo Test (2-3 minutes)
```bash
python examples/example.py
```
- Trains for 5 epochs
- Expected accuracy: ~96%
- Creates basic visualizations

#### 2. Full Training Test (3-5 minutes)
```bash
python src/mnist_classifier.py
```
- Complete 20-epoch training
- Expected accuracy: ~97-98%
- Generates all visualizations and saves model

#### 3. Custom Image Test
```bash
python examples/classify_custom_images.py
```
- Creates sample digit images
- Tests custom image classification
- Demonstrates batch processing

### Performance Benchmarks

| Hardware | Training Time (20 epochs) | Final Accuracy | Memory Usage |
|----------|---------------------------|----------------|--------------|
| CPU Only | 3-5 minutes | 97-98% | ~2-4 GB RAM |
| GPU (if available) | 1-2 minutes | 97-98% | ~2-4 GB RAM |

## 🛠️ Customization Options

### Model Architecture
```python
# Deeper network
classifier.build_model(hidden_layers=[512, 256, 128, 64], dropout_rate=0.3)

# Wider network  
classifier.build_model(hidden_layers=[1024, 512], dropout_rate=0.4)

# Minimal network for testing
classifier.build_model(hidden_layers=[32, 16], dropout_rate=0.1)
```

### Training Parameters
```python
# Longer training with smaller batches
classifier.train_model(epochs=50, batch_size=32, validation_split=0.15)

# Quick training for testing
classifier.train_model(epochs=5, batch_size=256)
```

### Image Processing
```python
# Custom image extensions
results = classifier.classify_images_in_directory(
    "images/", 
    image_extensions=['*.png', '*.tif', '*.bmp']
)

# Process single image with custom preprocessing
processed = classifier.preprocess_custom_image("my_digit.jpg")
```

## 🔧 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Solution: Ensure virtual environment is activated and dependencies installed
source venv/bin/activate
pip install -r requirements.txt
```

#### Low Accuracy (<90%)
- **Cause**: Training interrupted or insufficient epochs
- **Solution**: Increase epochs or check for early stopping
- **Custom Images**: Ensure images are clean, centered digits

#### Memory Errors
```python
# Reduce batch size
classifier.train_model(epochs=20, batch_size=32)  # instead of 128
```

#### Slow Training
- **CPU**: Normal, 3-5 minutes expected
- **Speed up**: Reduce model size, use fewer epochs, or enable GPU

#### Custom Image Classification Issues
- **Poor Results**: Check image quality, ensure single digits, proper contrast
- **Format Errors**: Verify supported formats (PNG, JPG, JPEG, BMP, TIFF)
- **Size Issues**: Images are automatically resized to 28x28

#### Plot Display Issues
- **Headless Systems**: Plots automatically save to assets/ directory
- **Display Problems**: Check matplotlib backend configuration

### Performance Optimization

#### For Speed
```python
# Reduce model complexity
classifier.build_model(hidden_layers=[64, 32], dropout_rate=0.1)

# Use larger batch sizes (if memory allows)
classifier.train_model(batch_size=256)

# Fewer epochs for quick testing
classifier.train_model(epochs=10)
```

#### For Accuracy
```python
# Increase model complexity
classifier.build_model(hidden_layers=[256, 128, 64, 32], dropout_rate=0.3)

# More training epochs
classifier.train_model(epochs=50)

# Smaller batch size for better gradient updates
classifier.train_model(batch_size=32)
```

## 📋 Validation Checklist

Before considering the system fully functional:

- [ ] ✅ Basic tests pass (4/4)
- [ ] ✅ MNIST training completes successfully  
- [ ] ✅ Test accuracy > 95%
- [ ] ✅ Model saves and loads correctly
- [ ] ✅ Visualizations generate properly
- [ ] ✅ Custom image classification works
- [ ] ✅ Batch processing functions correctly
- [ ] ✅ All example scripts run without errors

## 🔄 Development Workflow

### Adding New Features
```bash
# Create feature branch
git checkout -b feature/new-enhancement

# Make changes and test
python tests/test_basic.py

# Commit changes
git add .
git commit -m "Add new feature"

# Merge back to main
git checkout main
git merge feature/new-enhancement
```

### Model Experimentation
```python
# Try different architectures
for layers in [[64, 32], [128, 64], [256, 128, 64]]:
    classifier = MNISTClassifier(f"model_{len(layers)}_layers")
    classifier.load_and_preprocess_data()
    classifier.build_model(hidden_layers=layers)
    classifier.train_model(epochs=10)
    accuracy, _, _ = classifier.evaluate_model()
    print(f"Architecture {layers}: {accuracy:.3f} accuracy")
```

## 🚀 Advanced Usage

### Batch Processing Pipeline
```python
import glob
from src.mnist_classifier import MNISTClassifier

# Load model once
classifier = MNISTClassifier()
classifier.load_model("models/mnist_model.keras")

# Process multiple directories
directories = ["batch1/", "batch2/", "batch3/"]
all_results = {}

for directory in directories:
    print(f"Processing {directory}...")
    results = classifier.classify_images_in_directory(directory)
    all_results[directory] = results

# Generate comprehensive report
for directory, results in all_results.items():
    print(f"\n{directory}: {len(results)} images processed")
```

### Integration with Other Systems
```python
# REST API integration example
def classify_uploaded_image(image_bytes):
    # Save temporary image
    with open("temp_image.png", "wb") as f:
        f.write(image_bytes)
    
    # Classify
    digit, confidence = classifier.classify_image("temp_image.png")
    
    # Clean up
    os.remove("temp_image.png")
    
    return {"digit": int(digit), "confidence": float(confidence)}
```

## 📦 Dependencies

Core requirements:
- **TensorFlow** ≥2.15.0 - Deep learning framework
- **NumPy** ≥1.24.0 - Numerical computations
- **Matplotlib** ≥3.7.0 - Plotting and visualization
- **Scikit-learn** ≥1.3.0 - Machine learning metrics
- **Seaborn** ≥0.12.0 - Statistical visualization
- **OpenCV** - Image processing (auto-installed with tensorflow)
- **Pillow** - Additional image format support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`python tests/test_basic.py`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MNIST Dataset**: Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- **TensorFlow Team**: For the excellent deep learning framework
- **OpenCV Contributors**: For robust image processing capabilities

---

## 🎉 Ready to Use!

Your enhanced MNIST classifier is production-ready and supports both dataset training and custom image classification. The system has been thoroughly tested and includes comprehensive examples for all use cases.

**Next Steps:**
1. Train your model with `python examples/example.py`
2. Test with your own images using `python examples/classify_custom_images.py`
3. Integrate into your own projects using the flexible API
4. Experiment with different architectures and hyperparameters

Happy classifying! 🚀