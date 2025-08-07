"""
Simple example of using the MNIST classifier
"""

from mnist_classifier import MNISTClassifier

def simple_example():
    """Simple example showing basic usage."""
    print("MNIST Classification Example")
    print("=" * 40)
    
    # Create classifier
    classifier = MNISTClassifier("example_model")
    
    # Load data
    classifier.load_and_preprocess_data()
    
    # Build a simple model
    classifier.build_model(hidden_layers=[64, 32], dropout_rate=0.2)
    
    # Train for fewer epochs for quick demo
    classifier.train_model(epochs=5, batch_size=128)
    
    # Evaluate
    test_accuracy, y_pred, y_true = classifier.evaluate_model()
    
    # Show some visualizations
    classifier.plot_training_history()
    classifier.visualize_predictions(num_samples=5)
    
    print(f"\nQuick Demo Completed! Test Accuracy: {test_accuracy:.3f}")

if __name__ == "__main__":
    simple_example()