import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from main import test_model_simple

def quick_test():
    """Quick test using the test_model_simple function from main.py"""
    
    # Test data path
    test_data_path = 'Data/test/image'
    
    # Model configurations to test
    model_configs = [
        {'name': 'U-Net 2 blocks', 'path': 'best_u-net_2_blocks.pth'},
        {'name': 'U-Net 3 blocks', 'path': 'best_u-net_3_blocks.pth'},
        {'name': 'U-Net 4 blocks', 'path': 'best_u-net_4_blocks.pth'}
    ]
    
    print("Starting quick test of pretrained models...")
    print("=" * 50)
    
    for config in model_configs:
        model_name = config['name']
        model_path = config['path']
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
            
        print(f"\nTesting {model_name}...")
        
        # Use the simple test function
        predictions = test_model_simple(model_path, test_data_path)
        
        if predictions is not None:
            print(f"✓ Successfully generated predictions for {model_name}")
            print(f"  Prediction shape: {predictions.shape}")
            
            # Save a sample prediction visualization
            save_sample_prediction(predictions, model_name)
        else:
            print(f"✗ Failed to generate predictions for {model_name}")
    
    print("\n" + "=" * 50)
    print("Quick test completed!")

def save_sample_prediction(predictions, model_name, num_samples=20, cols=4):
    """Save sample prediction visualizations in a grid format"""
    os.makedirs('quick_test_results', exist_ok=True)
    
    # Convert predictions to numpy
    pred_np = predictions.squeeze().numpy()
    
    # Limit to the requested number of samples
    num_samples = min(num_samples, len(pred_np))
    
    # Calculate grid size
    rows = (num_samples + cols - 1) // cols  # Ceiling division
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = np.array(axes).reshape(-1)  # Flatten in case of single row
    
    # Plot each sample
    for i in range(num_samples):
        axes[i].imshow(pred_np[i], cmap='gray')
        axes[i].set_title(f'Sample {i+1}', fontsize=8)
        axes[i].axis('off')
    
    # Turn off unused axes
    for j in range(num_samples, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    filename = f"quick_test_results/{model_name.replace(' ', '_').lower()}_samples.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved sample predictions to: {filename}")

if __name__ == "__main__":
    # Test all available models
    quick_test()
    