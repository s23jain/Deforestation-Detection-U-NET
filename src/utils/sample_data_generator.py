"""
Sample Data Generator for Deforestation Detection
Creates realistic synthetic forest/deforestation data for initial testing
Version: 1.1 - Fixed JSON serialization issue
Authors: [Your Names]
Date: August 2025
"""

import numpy as np
import os
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

class ForestDataGenerator:
    """
    Generate synthetic forest and deforestation data
    """
    
    def __init__(self, output_dir='data/processed', image_size=128, seed=42):
        """
        Initialize data generator
        
        Args:
            output_dir (str): Output directory for generated data
            image_size (int): Size of generated images
            seed (int): Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.image_size = image_size
        self.seed = seed
        
        # Create output directories
        self.images_dir = os.path.join(output_dir, 'images')
        self.masks_dir = os.path.join(output_dir, 'masks')
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)
        
        # Set random seed
        np.random.seed(seed)
        
        # Define colors
        self.forest_color = [34, 139, 34]      # Forest Green
        self.deforested_color = [139, 69, 19]   # Saddle Brown
        self.water_color = [65, 105, 225]       # Royal Blue
        self.urban_color = [105, 105, 105]      # Dim Gray
    
    def generate_forest_pattern(self):
        """
        Generate realistic forest/deforestation patterns
        
        Returns:
            Binary mask where 1=forest, 0=deforested
        """
        # Start with random noise
        pattern = np.random.random((self.image_size, self.image_size))
        
        # Create forest patches using threshold
        forest_threshold = np.random.uniform(0.3, 0.7)
        forest_mask = pattern > forest_threshold
        
        # Add some circular deforested areas (logging patches)
        num_clearings = np.random.randint(2, 6)
        
        for _ in range(num_clearings):
            # Random clearing center and size
            center_x = np.random.randint(20, self.image_size - 20)
            center_y = np.random.randint(20, self.image_size - 20)
            radius = np.random.randint(10, 25)
            
            # Create circular mask
            y, x = np.ogrid[:self.image_size, :self.image_size]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            # Remove forest in this area
            forest_mask[mask] = False
        
        return forest_mask.astype(np.uint8)
    
    def add_realistic_features(self, image, forest_mask):
        """
        Add realistic features like water bodies, roads, etc.
        
        Args:
            image (np.array): Input image
            forest_mask (np.array): Forest mask
            
        Returns:
            Enhanced image with realistic features
        """
        # Add water bodies
        if np.random.random() > 0.7:  # 30% chance of water
            water_size = np.random.randint(5, 15)
            water_x = np.random.randint(0, self.image_size - water_size)
            water_y = np.random.randint(0, self.image_size - water_size)
            
            image[water_y:water_y+water_size, water_x:water_x+water_size] = self.water_color
            forest_mask[water_y:water_y+water_size, water_x:water_x+water_size] = 0
        
        # Add roads (thin lines)
        if np.random.random() > 0.8:  # 20% chance of roads
            if np.random.random() > 0.5:  # Horizontal road
                road_y = np.random.randint(10, self.image_size - 10)
                image[road_y:road_y+2, :] = self.urban_color
                forest_mask[road_y:road_y+2, :] = 0
            else:  # Vertical road
                road_x = np.random.randint(10, self.image_size - 10)
                image[:, road_x:road_x+2] = self.urban_color
                forest_mask[:, road_x:road_x+2] = 0
        
        return image, forest_mask
    
    def add_noise_and_blur(self, image):
        """
        Add realistic noise and blur to make images more satellite-like
        
        Args:
            image (np.array): Input image
            
        Returns:
            Enhanced image with noise and blur
        """
        # Convert to PIL for filtering
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        # Add slight blur to simulate atmospheric effects
        if np.random.random() > 0.5:
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Convert back to numpy
        image = np.array(pil_image)
        
        # Add random noise
        noise_level = np.random.randint(10, 30)
        noise = np.random.randint(-noise_level, noise_level, image.shape)
        image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def generate_single_sample(self, sample_id):
        """
        Generate a single image-mask pair
        
        Args:
            sample_id (int): Unique identifier for the sample
            
        Returns:
            tuple: (image_path, mask_path)
        """
        # Generate forest pattern
        forest_mask = self.generate_forest_pattern()
        
        # Create base image
        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Fill forest and deforested areas
        image[forest_mask == 1] = self.forest_color
        image[forest_mask == 0] = self.deforested_color
        
        # Add realistic features
        image, forest_mask = self.add_realistic_features(image, forest_mask)
        
        # Add noise and blur
        image = self.add_noise_and_blur(image)
        
        # Create binary mask (255 for forest, 0 for non-forest)
        mask = (forest_mask * 255).astype(np.uint8)
        
        # Save files
        image_filename = f'forest_sample_{sample_id:04d}.png'
        mask_filename = f'forest_mask_{sample_id:04d}.png'
        
        image_path = os.path.join(self.images_dir, image_filename)
        mask_path = os.path.join(self.masks_dir, mask_filename)
        
        Image.fromarray(image).save(image_path)
        Image.fromarray(mask).save(mask_path)
        
        return image_path, mask_path
    
    def generate_dataset(self, num_samples=100, show_progress=True):
        """
        Generate complete dataset
        
        Args:
            num_samples (int): Number of samples to generate
            show_progress (bool): Whether to show progress bar
            
        Returns:
            dict: Dataset statistics
        """
        print(f"🌳 Generating {num_samples} synthetic forest samples...")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🖼️  Image size: {self.image_size}x{self.image_size}")
        
        # Generate samples with progress bar
        sample_info = []
        
        iterator = tqdm(range(num_samples), desc="Generating") if show_progress else range(num_samples)
        
        for i in iterator:
            image_path, mask_path = self.generate_single_sample(i)
            sample_info.append({
                'id': i,
                'image_path': image_path,
                'mask_path': mask_path
            })
        
        # Calculate dataset statistics
        total_pixels = num_samples * self.image_size * self.image_size
        forest_pixels = 0
        
        for info in sample_info:
            mask = np.array(Image.open(info['mask_path']))
            forest_pixels += np.sum(mask > 0)
        
        # FIXED: Convert numpy types to Python native types for JSON serialization
        forest_pixels = int(forest_pixels)
        total_pixels = int(total_pixels)
        forest_percentage = float((forest_pixels / total_pixels) * 100)
        
        # Save dataset info
        dataset_info = {
            'generation_date': datetime.now().isoformat(),
            'num_samples': int(num_samples),
            'image_size': int(self.image_size),
            'total_pixels': total_pixels,
            'forest_pixels': forest_pixels,
            'forest_percentage': forest_percentage,
            'seed': int(self.seed),
            'samples': sample_info
        }
        
        info_path = os.path.join(self.output_dir, 'dataset_info.json')
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\n✅ Dataset generation complete!")
        print(f"   📊 Samples: {num_samples}")
        print(f"   🌲 Forest coverage: {forest_percentage:.1f}%")
        print(f"   💾 Dataset info saved to: {info_path}")
        
        return dataset_info
    
    def visualize_samples(self, num_samples=6):
        """
        Create visualization of generated samples
        
        Args:
            num_samples (int): Number of samples to visualize
        """
        print(f"📊 Creating visualization of {num_samples} samples...")
        
        fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
        
        for i in range(num_samples):
            # Load image and mask
            image_path = os.path.join(self.images_dir, f'forest_sample_{i:04d}.png')
            mask_path = os.path.join(self.masks_dir, f'forest_mask_{i:04d}.png')
            
            if os.path.exists(image_path) and os.path.exists(mask_path):
                image = Image.open(image_path)
                mask = Image.open(mask_path)
                
                # Show image
                axes[0, i].imshow(image)
                axes[0, i].set_title(f'Sample {i+1}\n(Satellite Image)')
                axes[0, i].axis('off')
                
                # Show mask
                axes[1, i].imshow(mask, cmap='gray')
                axes[1, i].set_title(f'Forest Mask {i+1}\n(White=Forest)')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join('results/figures', 'sample_data_visualization.png')
        os.makedirs('results/figures', exist_ok=True)
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualization saved to: {viz_path}")

# CLI interface
if __name__ == "__main__":
    print("🌲 Forest Data Generator v1.1")
    print("=" * 50)
    
    # Create generator
    generator = ForestDataGenerator(
        output_dir='data/processed',
        image_size=128,
        seed=42
    )
    
    # Generate dataset
    dataset_info = generator.generate_dataset(num_samples=100)
    
    # Create visualization
    generator.visualize_samples(num_samples=6)
    
    print("\n🎉 Data generation complete! Ready for model training.")
