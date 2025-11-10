from torch.utils.data import DataLoader 
import torch.optim as optim
from tqdm import tqdm 
from get_img_mask import SegmentationDataset
from unet import UNet
import torch 
import matplotlib.pyplot as plt 
import torchvision.transforms as T 


num_classes = 4 
batch_size = 8 
epochs = 30 
learning_rate = 1e-4 

transform = T.Compose([
    T.Resize((256,256)),
    T.ToTensor()
]
)

mask_dir = "C:\Local\dataset\MaSTr1325_masks_512x384" 
img_dir = "C:\Local\dataset\MaSTr1325_images_512x384"

train_dataset = SegmentationDataset(img_dir,mask_dir, num_classes, transform=transform )
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True) 

device = torch.device("cude" if torch.cuda.is_available() else "cpu") 
model = UNet(in_channels =3 , num_classes = num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = torch.nn.CrossEntropyLoss()

model.train() 
for epoch in range(epochs):
    epoch_loss = 0 
    for images, masks in tqdm(train_loader, desc= f"Epoch {epoch+1}/{epochs}"): 
        images = images.to(device) 
        masks = masks.to(device) 

        optimizer.zero_grad()
        outputs = model(images) 
        loss = criterion(outputs,masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()  

    print(  f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}") 


def predict(model, image_path, device):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = T.Resize((256, 256))(image)
    image_tensor = T.ToTensor()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        return np.array(image), pred_mask

original, pred = predict(model, "test_img.png", device)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(original)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(pred, cmap="tab10")  # Use colormap for multi-class
plt.axis("off")
plt.show()