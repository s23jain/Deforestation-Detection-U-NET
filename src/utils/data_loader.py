"""
Data Loading and Preprocessing Utilities
Handles loading of synthetic and real satellite imagery data
"""

import os
import numpy as np
import cv2
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm

class ForestDataLoader:
    """
    Data loader for forest/deforestation detection datasets
    """
    
    def __init__(self, data_dir='data/processed', image_size=128, batch_size=8):
        """
        Initialize data loader
        
        Args:
            data_dir (str): Directory containing images and masks
            image_size (int): Target size for images
            batch_size (int): Batch size for training
        """
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        self.masks_dir = os.path.join(data_dir, 'masks')
        self.image_size = image_size
        self.batch_size = batch_size
        
        # Load dataset info if available
        info_path = os.path.join(data_dir, 'dataset_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self.dataset_info = json.load(f)
        else:
            self.dataset_info = None
    
    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess a single image
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            Preprocessed image array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def load_and_preprocess_mask(self, mask_path):
        """
        Load and preprocess a single mask
        
        Args:
            mask_path (str): Path to mask file
            
        Returns:
            Preprocessed mask array
        """
        # Load mask in grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        # Resize if needed
        if mask.shape != (self.image_size, self.image_size):
            mask = cv2.resize(mask, (self.image_size, self.image_size))
        
        # Normalize to [0, 1] and add channel dimension
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=-1)
        
        return mask
    
    def load_dataset(self, max_samples=None):
        """
        Load complete dataset
        
        Args:
            max_samples (int): Maximum number of samples to load
            
        Returns:
            tuple: (images, masks) arrays
        """
        # Get list of image files
        image_files = sorted([f for f in os.listdir(self.images_dir) 
                             if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if max_samples:
            image_files = image_files[:max_samples]
        
        print(f"📁 Loading {len(image_files)} samples from {self.data_dir}")
        
        images = []
        masks = []
        
        for filename in tqdm(image_files, desc="Loading data"):
            try:
                # Construct paths
                image_path = os.path.join(self.images_dir, filename)
                # Assume mask has same name but in masks directory
                mask_filename = filename.replace('sample', 'mask').replace('forest_sample', 'forest_mask')
                mask_path = os.path.join(self.masks_dir, mask_filename)
                
                # Load and preprocess
                image = self.load_and_preprocess_image(image_path)
                mask = self.load_and_preprocess_mask(mask_path)
                
                images.append(image)
                masks.append(mask)
                
            except Exception as e:
                print(f"Warning: Could not load {filename}: {str(e)}")
                continue
        
        images = np.array(images)
        masks = np.array(masks)
        
        print(f"✅ Loaded dataset:")
        print(f"   Images shape: {images.shape}")
        print(f"   Masks shape: {masks.shape}")
        print(f"   Memory usage: {(images.nbytes + masks.nbytes) / 1024**2:.1f} MB")
        
        return images, masks
    
    def create_train_val_test_split(self, test_size=0.2, val_size=0.2, random_state=42):
        """
        Create train/validation/test splits
        
        Args:
            test_size (float): Fraction for test set
            val_size (float): Fraction for validation set (from remaining after test)
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Load data
        images, masks = self.load_dataset()
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, masks, test_size=test_size, random_state=random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        print(f"📊 Data split summary:")
        print(f"   Training:   {len(X_train):3d} samples ({len(X_train)/len(images)*100:.1f}%)")
        print(f"   Validation: {len(X_val):3d} samples ({len(X_val)/len(images)*100:.1f}%)")
        print(f"   Test:       {len(X_test):3d} samples ({len(X_test)/len(images)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_tf_dataset(self, images, masks, shuffle=True, augment=False):
        """
        Create TensorFlow dataset for efficient training
        
        Args:
            images (np.array): Image data
            masks (np.array): Mask data
            shuffle (bool): Whether to shuffle data
            augment (bool): Whether to apply data augmentation
            
        Returns:
            tf.data.Dataset: TensorFlow dataset
        """
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((images, masks))
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(images))
        
        # Apply augmentation if requested
        if augment:
            dataset = dataset.map(self._augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Batch and prefetch
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _augment_data(self, image, mask):
        """
        Apply data augmentation
        
        Args:
            image: Input image tensor
            mask: Input mask tensor
            
        Returns:
            tuple: (augmented_image, augmented_mask)
        """
        # Random horizontal flip
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        
        # Random vertical flip
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)
        
        # Random rotation (90, 180, 270 degrees)
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k)
        mask = tf.image.rot90(mask, k)
        
        return image, mask

# Test the data loader
if __name__ == "__main__":
    print("🔍 Testing Data Loader...")
    print("=" * 50)
    
    # Create data loader
    loader = ForestDataLoader(
        data_dir='data/processed',
        image_size=128,
        batch_size=8
    )
    
    # Test data loading
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = loader.create_train_val_test_split()
        
        # Create TensorFlow datasets
        train_ds = loader.create_tf_dataset(X_train, y_train, shuffle=True, augment=True)
        val_ds = loader.create_tf_dataset(X_val, y_val, shuffle=False, augment=False)
        
        print(f"\n✅ Data loader test successful!")
        print(f"   TensorFlow datasets created")
        print(f"   Ready for model training")
        
    except Exception as e:
        print(f"❌ Data loader test failed: {str(e)}")
