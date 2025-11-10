import os 
import torch 
import numpy as np 
from torch.utils.data import Dataset 
from PIL import Image 
import torchvision.transforms as T 

class SegmentationDataset(Dataset): 
    def __init__(self, image_dir, mask_dir, transform = None): 
        self.image_dir = image_dir 
        self.mask_dir = mask_dir 
        self.transform = transform 
        self.num_classes = num_classes 
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self): 
        return len(self.images) 
    
    def __getitem(self, idx): 
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.mask[idx])
        
        image = Image.open(img_path).convert("RGB") 
        mask  = Image.open(mask_path).convert("L")

        if self.transform: 
            seed = torch.randint(0,2**32, (1,)).item() 
            torch.manual_seed(seed) 
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask) 
        
        else: 
            image = T.ToTensor()(image) 
            #mask = T.ToTensor()(mask) 
            mask = torch.from_numpy(np.array(mask)).long() 
    
        return image, mask
            