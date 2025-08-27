"""
Test script for model creation and basic functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.simple_unet import create_simple_unet
import tensorflow as tf
import numpy as np

def test_model_creation():
    """Test basic model creation and functionality"""
    print("🧪 Testing Model Creation...")
    print("=" * 50)
    
    try:
        # Create model
        model = create_simple_unet(input_height=128, input_width=128, channels=3)
        print("✅ Model creation: SUCCESS")
        
        # Test model compilation
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Model compilation: SUCCESS")
        
        # Test with dummy data
        batch_size = 4
        dummy_images = np.random.random((batch_size, 128, 128, 3))
        dummy_masks = np.random.randint(0, 2, (batch_size, 128, 128, 1)).astype(np.float32)
        
        # Test prediction
        predictions = model.predict(dummy_images, verbose=0)
        print("✅ Model prediction: SUCCESS")
        
        # Test training step
        history = model.fit(dummy_images, dummy_masks, epochs=1, verbose=0)
        print("✅ Model training step: SUCCESS")
        
        # Print summary
        print(f"\n📊 Model Summary:")
        print(f"   Total parameters: {model.count_params():,}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_model_creation()
    
    if success:
        print("\n🎉 All tests passed! Ready to proceed with data generation.")
    else:
        print("\n💥 Tests failed! Please check the error messages above.")
