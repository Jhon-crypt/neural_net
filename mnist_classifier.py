"""
MNIST Handwritten Digit Classification using Neural Networks
============================================================

This script implements a neural network for classifying handwritten digits
from the MNIST dataset using TensorFlow/Keras.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os


class MNISTClassifier:
    """A neural network classifier for MNIST handwritten digits."""
    
    def __init__(self, model_name="mnist_model"):
        self.model = None
        self.model_name = model_name
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the MNIST dataset."""
        print("Loading MNIST dataset...")
        
        # Load the MNIST dataset
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize pixel values to [0, 1] range
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape data to flatten images (28x28 -> 784)
        x_train = x_train.reshape(x_train.shape[0], 28 * 28)
        x_test = x_test.reshape(x_test.shape[0], 28 * 28)
        
        # Convert labels to categorical (one-hot encoding)
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        print(f"Training data shape: {x_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
    def build_model(self, hidden_layers=[128, 64], dropout_rate=0.2):
        """
        Build a neural network model.
        
        Args:
            hidden_layers (list): List of neurons in each hidden layer
            dropout_rate (float): Dropout rate for regularization
        """
        print("Building neural network model...")
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(hidden_layers[0], 
                              activation='relu', 
                              input_shape=(784,)))
        model.add(layers.Dropout(dropout_rate))
        
        # Hidden layers
        for neurons in hidden_layers[1:]:
            model.add(layers.Dense(neurons, activation='relu'))
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer (10 classes for digits 0-9)
        model.add(layers.Dense(10, activation='softmax'))
        
        # Compile the model
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        self.model = model
        
        # Print model summary
        print("\nModel Architecture:")
        model.summary()
        
    def train_model(self, epochs=20, batch_size=128, validation_split=0.1):
        """
        Train the neural network model.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print(f"\nTraining model for {epochs} epochs...")
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        
    def evaluate_model(self):
        """Evaluate the model on test data and print metrics."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print("\nEvaluating model on test data...")
        
        # Evaluate on test data
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Make predictions
        y_pred = self.model.predict(self.x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes))
        
        return test_accuracy, y_pred_classes, y_true_classes
        
    def plot_training_history(self):
        """Plot training and validation metrics."""
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def visualize_predictions(self, num_samples=10):
        """Visualize sample predictions."""
        # Get original images for visualization
        (x_train_orig, _), (x_test_orig, _) = keras.datasets.mnist.load_data()
        
        # Make predictions
        predictions = self.model.predict(self.x_test[:num_samples], verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(self.y_test[:num_samples], axis=1)
        
        # Plot samples
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(num_samples):
            axes[i].imshow(x_test_orig[i], cmap='gray')
            axes[i].set_title(f'True: {true_classes[i]}, Pred: {predicted_classes[i]}')
            axes[i].axis('off')
            
            # Color the title based on correctness
            if true_classes[i] == predicted_classes[i]:
                axes[i].title.set_color('green')
            else:
                axes[i].title.set_color('red')
        
        plt.tight_layout()
        plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_model(self, filepath=None):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        if filepath is None:
            filepath = f"{self.model_name}.keras"
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        """Load a saved model."""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


def main():
    """Main function to run the MNIST classification."""
    # Create classifier instance
    classifier = MNISTClassifier()
    
    # Load and preprocess data
    classifier.load_and_preprocess_data()
    
    # Build model
    classifier.build_model(hidden_layers=[128, 64], dropout_rate=0.2)
    
    # Train model
    classifier.train_model(epochs=20, batch_size=128)
    
    # Evaluate model
    test_accuracy, y_pred, y_true = classifier.evaluate_model()
    
    # Visualize results
    classifier.plot_training_history()
    classifier.plot_confusion_matrix(y_true, y_pred)
    classifier.visualize_predictions()
    
    # Save model
    classifier.save_model()
    
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()