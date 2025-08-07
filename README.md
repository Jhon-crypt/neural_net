# MNIST Handwritten Digit Classification

A neural network implementation for classifying handwritten digits using the MNIST dataset with TensorFlow/Keras.

## Overview

This project implements a deep neural network to classify handwritten digits (0-9) from the famous MNIST dataset. The model achieves high accuracy using a multi-layer perceptron architecture with dropout regularization.

## Features

- **Data Preprocessing**: Automatic loading and normalization of MNIST dataset
- **Flexible Architecture**: Configurable hidden layers and dropout rates
- **Training Optimization**: Early stopping and learning rate reduction callbacks
- **Comprehensive Evaluation**: Accuracy metrics, classification reports, and confusion matrix
- **Visualization**: Training history plots, confusion matrix, and sample predictions
- **Model Persistence**: Save and load trained models

## Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Dependencies
- TensorFlow 2.15.0
- NumPy 1.24.3
- Matplotlib 3.7.1
- Scikit-learn 1.3.0
- Seaborn 0.12.2

## Usage

### Quick Start

Run the complete training pipeline:

```bash
python mnist_classifier.py
```

This will:
1. Load and preprocess the MNIST dataset
2. Build a neural network model
3. Train the model with validation
4. Evaluate performance on test data
5. Generate visualization plots
6. Save the trained model

### Custom Usage

```python
from mnist_classifier import MNISTClassifier

# Create classifier
classifier = MNISTClassifier()

# Load and preprocess data
classifier.load_and_preprocess_data()

# Build custom model architecture
classifier.build_model(hidden_layers=[256, 128, 64], dropout_rate=0.3)

# Train with custom parameters
classifier.train_model(epochs=30, batch_size=64)

# Evaluate and visualize
test_accuracy, y_pred, y_true = classifier.evaluate_model()
classifier.plot_training_history()
classifier.plot_confusion_matrix(y_true, y_pred)
classifier.visualize_predictions(num_samples=20)

# Save the model
classifier.save_model("my_mnist_model.h5")
```

## Model Architecture

The default neural network architecture includes:

- **Input Layer**: 784 neurons (28×28 flattened pixels)
- **Hidden Layer 1**: 128 neurons with ReLU activation + Dropout (0.2)
- **Hidden Layer 2**: 64 neurons with ReLU activation + Dropout (0.2)
- **Output Layer**: 10 neurons with Softmax activation (for 10 digit classes)

## Training Details

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

## Expected Results

The model typically achieves:
- **Training Accuracy**: ~99%
- **Test Accuracy**: ~97-98%
- **Training Time**: 2-5 minutes (depending on hardware)

## Output Files

After training, the following files are generated:
- `mnist_model.h5`: Saved trained model
- `training_history.png`: Training and validation metrics plot
- `confusion_matrix.png`: Confusion matrix heatmap
- `sample_predictions.png`: Sample predictions visualization

## Project Structure

```
neural_net/
├── mnist_classifier.py    # Main classifier implementation
├── requirements.txt       # Project dependencies
├── README.md             # Project documentation
└── example.py            # Simple usage example
```

## Customization Options

### Model Architecture
```python
# Deeper network
classifier.build_model(hidden_layers=[512, 256, 128, 64], dropout_rate=0.3)

# Wider network
classifier.build_model(hidden_layers=[1024, 512], dropout_rate=0.4)
```

### Training Parameters
```python
# Longer training with smaller batches
classifier.train_model(epochs=50, batch_size=32, validation_split=0.15)
```

## Performance Tips

1. **GPU Acceleration**: Install `tensorflow-gpu` for faster training
2. **Batch Size**: Increase batch size if you have sufficient memory
3. **Learning Rate**: Use learning rate scheduling for better convergence
4. **Regularization**: Adjust dropout rate based on overfitting behavior

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce batch size
2. **Slow Training**: Enable GPU or reduce model complexity
3. **Poor Accuracy**: Increase epochs or adjust learning rate
4. **Overfitting**: Increase dropout rate or add more regularization

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for improvements.