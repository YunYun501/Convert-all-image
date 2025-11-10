import matplotlib.pyplot as plt 
import numpy as np 
import torchvision.transforms as T

def predict(model, image_path, device):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = T.Resize((256, 256))(image)
    image_tensor = T.ToTensor()(image).unsqueeze(0).to(device)

