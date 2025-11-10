import torch
import torch.nn as nn 
import torch.nn.functional as F 

class DoubleConv(nn.Module): 
    def __init__(self, in_ch,out_ch):
        super().__init__()
        self.conv = nn.sequential(
            nn.Conv2d(in_ch,out_ch, 3,padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3,padding=1, bias = False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True)
        )
    
    def forward(self, x):
        return self.conv(x) 
    
class UNet(nn.Module): 
    def __init__(self, in_channels = 3, num_classes =4): 
        self.down1 = DoubleConv(in_channels,64)
        self.down2 = DoubleConv(64,128) 
        self.down3 = DoubleConv(128,256) 
        self.down4 = DoubleConv(256,512) 
        self.bottleneck = DoubleConv(512,1024) 

        self.upconv4 = nn.ConvTranspose2d(1024,512, 2, stride = 2)
        self.up4 = DoubleConv(1024,512) 
        self.upconv3 = nn.ConvTranspose2d(512,256, 2, stride =2) 
        self.up3 = DoubleConv(512,256) 
        self.upconv2 = nn.ConvTranspose2d(256,128, 2, stride =2) 
        self.up2 = DoubleConv(256,128) 
        self.upconv1 = nn.ConvTransposed2d(128,64, 2, stride =2) 
        self.up1 = DoubleConv(128,64)

        self.final_conv = nn.Conv2d(64,num_classes, kernel_size=1)

    def forward(self, x): 
        x1 = self.down1(x) 
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))

        b = self.bottleneck(self.pool(x4)) 
        
        u4 = self.upconv4(b) 
        u4 = torch.cat([u4, x4], dim=1) 
        u4 = self.up4(u4) 

        u3 = self.upconv3(u4) 
        u3 = torch.cat([u3, x3],dim=1) 
        u3 = self.up3(u3) 

        u2 = self.upconv2(u3)
        u2 = torch.cat([u2,x2],dim=1) 
        u2 = self.up2(u2)

        u1 = self.upconv1(u2)
        u1 = torch.cat([u1,x1],dim=1) 
        u1 = self.up1(u1) 

        out = self.final_conv(u1) 


