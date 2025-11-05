"""
Image to Tensor conversion utility
"""
import torch
import numpy as np
from PIL import Image

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
    # Example usage
    # tensor = image_to_tensor("path/to/image.jpg")
    pass