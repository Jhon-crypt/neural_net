"""
Basic test to verify the MNIST classifier imports and basic functionality
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✓ Seaborn imported successfully")
    except ImportError as e:
        print(f"✗ Seaborn import failed: {e}")
        return False
    
    try:
        from mnist_classifier import MNISTClassifier
        print("✓ MNISTClassifier imported successfully")
    except ImportError as e:
        print(f"✗ MNISTClassifier import failed: {e}")
        return False
    
    return True

def test_data_loading():
    """Test MNIST data loading."""
    print("\nTesting MNIST data loading...")
    
    try:
        from mnist_classifier import MNISTClassifier
        classifier = MNISTClassifier()
        classifier.load_and_preprocess_data()
        
        print(f"✓ Training data shape: {classifier.x_train.shape}")
        print(f"✓ Test data shape: {classifier.x_test.shape}")
        print("✓ MNIST data loaded and preprocessed successfully")
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

def test_model_building():
    """Test model building."""
    print("\nTesting model building...")
    
    try:
        from mnist_classifier import MNISTClassifier
        classifier = MNISTClassifier()
        classifier.load_and_preprocess_data()
        classifier.build_model(hidden_layers=[32, 16], dropout_rate=0.2)
        
        print("✓ Model built successfully")
        print(f"✓ Model has {classifier.model.count_params()} parameters")
        return True
    except Exception as e:
        print(f"✗ Model building failed: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability."""
    print("\nTesting GPU availability...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✓ Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
        else:
            print("ℹ No GPU detected, will use CPU")
        return True
    except Exception as e:
        print(f"✗ GPU check failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("=" * 50)
    print("MNIST Classifier Basic Tests")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Loading Test", test_data_loading),
        ("Model Building Test", test_model_building),
        ("GPU Availability Test", test_gpu_availability)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python example.py' for a quick demo")
        print("2. Run 'python mnist_classifier.py' for full training")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)