"""
Example: Classify custom handwritten digit images
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from mnist_classifier import MNISTClassifier
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_sample_digits():
    """Create sample digit images for testing."""
    print("Creating sample digit images...")
    
    # Create assets directory if it doesn't exist
    os.makedirs('assets/sample_digits', exist_ok=True)
    
    # Create simple digit images
    digits_data = [
        # Simple representations that should be recognizable
        (0, [[0,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [0,1,1,1,0]]),
        (1, [[0,0,1,0,0], [0,1,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,1,1,1,0]]),
        (2, [[0,1,1,1,0], [1,0,0,0,1], [0,0,1,1,0], [0,1,0,0,0], [1,1,1,1,1]]),
        (3, [[1,1,1,1,0], [0,0,0,0,1], [0,1,1,1,0], [0,0,0,0,1], [1,1,1,1,0]]),
        (4, [[1,0,0,1,0], [1,0,0,1,0], [1,1,1,1,1], [0,0,0,1,0], [0,0,0,1,0]]),
    ]
    
    for digit, pattern in digits_data:
        # Create a larger image (140x140) for better quality
        img = Image.new('L', (140, 140), color=255)  # White background
        draw = ImageDraw.Draw(img)
        
        # Scale up the pattern
        cell_size = 20
        offset_x = (140 - len(pattern[0]) * cell_size) // 2
        offset_y = (140 - len(pattern) * cell_size) // 2
        
        for y, row in enumerate(pattern):
            for x, pixel in enumerate(row):
                if pixel == 1:
                    # Draw black rectangles for the digit
                    x1 = offset_x + x * cell_size
                    y1 = offset_y + y * cell_size
                    x2 = x1 + cell_size
                    y2 = y1 + cell_size
                    draw.rectangle([x1, y1, x2, y2], fill=0)  # Black
        
        # Save the image - create directory if it doesn't exist
        import os
        os.makedirs('assets/sample_digits', exist_ok=True)
        img.save(f'assets/sample_digits/digit_{digit}.png')
    
    print("Sample digits created in assets/sample_digits/")

def classify_single_image_example():
    """Example of classifying a single image."""
    print("\n" + "="*50)
    print("SINGLE IMAGE CLASSIFICATION EXAMPLE")
    print("="*50)
    
    # Load pre-trained model (you need to train it first)
    classifier = MNISTClassifier()
    
    try:
        classifier.load_model('models/mnist_model.keras')
        print("✅ Model loaded successfully!")
    except:
        print("❌ No trained model found. Please run the main training script first.")
        return
    
    # Create sample images if they don't exist
    if not os.path.exists('assets/sample_digits'):
        create_sample_digits()
    
    # Classify a single image
    image_path = 'assets/sample_digits/digit_2.png'
    if os.path.exists(image_path):
        print(f"\nClassifying image: {image_path}")
        digit, confidence = classifier.classify_image(image_path)
        print(f"Result: Digit {digit} with confidence {confidence:.3f}")
    else:
        print("Sample image not found. Creating sample images...")
        create_sample_digits()

def classify_directory_example():
    """Example of classifying all images in a directory."""
    print("\n" + "="*50)
    print("DIRECTORY CLASSIFICATION EXAMPLE")
    print("="*50)
    
    # Load pre-trained model
    classifier = MNISTClassifier()
    
    try:
        classifier.load_model('models/mnist_model.keras')
        print("✅ Model loaded successfully!")
    except:
        print("❌ No trained model found. Please run the main training script first.")
        return
    
    # Create sample images if they don't exist
    if not os.path.exists('assets/sample_digits'):
        create_sample_digits()
    
    # Classify all images in directory
    results = classifier.classify_images_in_directory('assets/sample_digits')
    
    # Show results summary
    print("\nDetailed Results:")
    for image_path, (digit, confidence) in results.items():
        filename = os.path.basename(image_path)
        print(f"{filename}: Digit {digit} (confidence: {confidence:.3f})")

def visualize_predictions_example():
    """Example of visualizing predictions."""
    print("\n" + "="*50)
    print("VISUALIZATION EXAMPLE")
    print("="*50)
    
    classifier = MNISTClassifier()
    
    try:
        classifier.load_model('models/mnist_model.keras')
        print("✅ Model loaded successfully!")
    except:
        print("❌ No trained model found. Please run the main training script first.")
        return
    
    # Create sample images if they don't exist
    if not os.path.exists('assets/sample_digits'):
        create_sample_digits()
    
    # Get list of sample images
    import glob
    image_files = glob.glob('assets/sample_digits/*.png')
    
    if image_files:
        print(f"Visualizing predictions for {len(image_files)} images...")
        classifier.visualize_custom_predictions(image_files)
        print("✅ Visualization saved to assets/custom_predictions.png")
    else:
        print("No images found for visualization.")

def main():
    """Run all custom image classification examples."""
    print("MNIST Custom Image Classification Examples")
    print("="*60)
    
    # Run examples
    classify_single_image_example()
    classify_directory_example() 
    visualize_predictions_example()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("\nTo classify your own images:")
    print("1. Place your digit images in a directory")
    print("2. Use classifier.classify_images_in_directory('your_directory')")
    print("3. Images should contain single handwritten digits")
    print("4. Supported formats: PNG, JPG, JPEG, BMP, TIFF")

if __name__ == "__main__":
    main()