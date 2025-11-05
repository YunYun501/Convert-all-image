"""
Image to Tensor conversion utility
"""
import torch
import numpy as np
from PIL import Image
import os

def image_to_tensor(image_path):
    """
    Convert an image to a PyTorch tensor
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        torch.Tensor: Converted tensor
    """
    # Open image
    image = Image.open(image_path)
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Convert to tensor
    tensor = torch.from_numpy(image_array)
    
    return tensor

if __name__ == "__main__":
    # Example usage with mask image
    mask_dir = "../dataset/MaSTr1325_masks_512x384"
    mask_file = "0001m.png"
    mask_path = os.path.join(mask_dir, mask_file)
    
    if os.path.exists(mask_path):
        tensor = image_to_tensor(mask_path)
        print(f"Converted {mask_file} to tensor")
        print(f"Tensor shape: {tensor.shape}")
        print(f"Tensor dtype: {tensor.dtype}")
        
        # Save full tensor to file
        output_path = "../output/tensor_output.txt"
        torch.set_printoptions(profile="full")  # Show full tensor
        with open(output_path, 'w') as f:
            f.write(f"Converted {mask_file} to tensor\n")
            f.write(f"Tensor shape: {tensor.shape}\n")
            f.write(f"Tensor dtype: {tensor.dtype}\n")
            f.write(f"Full tensor content:\n")
            f.write(str(tensor))
        
        print(f"Full tensor saved to {output_path}")
    else:
        print(f"File {mask_path} not found")