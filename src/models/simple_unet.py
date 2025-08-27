"""
Simple U-Net Implementation for Deforestation Detection
Authors: [Your Names]
Date: August 2025
Version: 1.0
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

class SimpleUNet:
    """
    A simplified U-Net implementation for binary segmentation
    Perfect for learning and initial experiments
    """
    
    def __init__(self, input_height=128, input_width=128, channels=3):
        """
        Initialize U-Net parameters
        
        Args:
            input_height (int): Height of input images
            input_width (int): Width of input images  
            channels (int): Number of input channels (3 for RGB)
        """
        self.input_height = input_height
        self.input_width = input_width
        self.channels = channels
        
    def conv_block(self, inputs, filters, kernel_size=3):
        """
        Standard convolution block with BatchNormalization
        
        Args:
            inputs: Input tensor
            filters (int): Number of filters
            kernel_size (int): Size of convolution kernel
            
        Returns:
            Processed tensor
        """
        x = layers.Conv2D(filters, kernel_size, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        return x
    
    def encoder_block(self, inputs, filters):
        """
        Encoder block: Conv -> Pool
        
        Args:
            inputs: Input tensor
            filters (int): Number of filters
            
        Returns:
            conv: Feature maps before pooling (for skip connections)
            pool: Pooled features
        """
        conv = self.conv_block(inputs, filters)
        pool = layers.MaxPooling2D(pool_size=(2, 2))(conv)
        return conv, pool
    
    def decoder_block(self, inputs, skip_features, filters):
        """
        Decoder block: Upsample -> Concatenate -> Conv
        
        Args:
            inputs: Input tensor from previous layer
            skip_features: Features from encoder (skip connection)
            filters (int): Number of filters
            
        Returns:
            Processed tensor
        """
        # Upsample
        up = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(inputs)
        
        # Concatenate with skip connection
        concat = layers.Concatenate()([up, skip_features])
        
        # Convolution
        conv = self.conv_block(concat, filters)
        
        return conv
    
    def build_model(self):
        """
        Build complete U-Net model
        
        Returns:
            Complete U-Net model
        """
        # Input layer
        inputs = layers.Input((self.input_height, self.input_width, self.channels))
        
        # Encoder Path (Contracting Path)
        e1, p1 = self.encoder_block(inputs, 32)    # 128x128 -> 64x64
        e2, p2 = self.encoder_block(p1, 64)        # 64x64 -> 32x32
        e3, p3 = self.encoder_block(p2, 128)       # 32x32 -> 16x16
        e4, p4 = self.encoder_block(p3, 256)       # 16x16 -> 8x8
        
        # Bottleneck (Bridge)
        b = self.conv_block(p4, 512)               # 8x8
        
        # Decoder Path (Expansive Path)  
        d4 = self.decoder_block(b, e4, 256)        # 8x8 -> 16x16
        d3 = self.decoder_block(d4, e3, 128)       # 16x16 -> 32x32
        d2 = self.decoder_block(d3, e2, 64)        # 32x32 -> 64x64
        d1 = self.decoder_block(d2, e1, 32)        # 64x64 -> 128x128
        
        # Output layer
        outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(d1)
        
        # Create model
        model = Model(inputs, outputs, name='Simple_UNet_v1.0')
        
        return model

def create_simple_unet(input_height=128, input_width=128, channels=3):
    """
    Factory function to create U-Net model
    
    Args:
        input_height (int): Height of input images
        input_width (int): Width of input images
        channels (int): Number of input channels
        
    Returns:
        Compiled U-Net model
    """
    unet = SimpleUNet(input_height, input_width, channels)
    model = unet.build_model()
    return model

# Test the model creation
if __name__ == "__main__":
    print("Testing Simple U-Net Model Creation...")
    print("=" * 50)
    
    # Create model
    model = create_simple_unet()
    
    # Display model summary
    model.summary()
    
    # Test with dummy data
    import numpy as np
    dummy_input = np.random.random((1, 128, 128, 3))
    output = model.predict(dummy_input, verbose=0)
    
    print(f"\n✅ Model Test Results:")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"   Model parameters: {model.count_params():,}")
    print("\n🎉 Simple U-Net model created successfully!")
