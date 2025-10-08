import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=21, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Upsampling path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final classification layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downward pass (Encoder)
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] 

        # Upward pass (Decoder)
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) # ConvTranspose2d (upsamples)
            skip_connection = skip_connections[idx // 2] # Get corresponding skip connection

            # Handle dimension mismatch
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            # Concatenate skip connection with upsampled feature map
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip) # DoubleConv (processes concatenated features)

        return self.final_conv(x) # Return raw logits

# Test with multi-class segmentation
def test():
    # Test with a 256x256 image
    x = torch.randn((2, 3, 256, 256))  # Batch of 2 RGB images, 256x256 resolution
    model = UNET(in_channels=3, out_channels=21) # 21 classes for PASCAL VOC
    preds = model(x)
    assert preds.shape == (2, 21, 256, 256), f"Expected (2,21,256,256), got {preds.shape}"
    print(f"Test passed! Output shape: {preds.shape}")

if __name__ == "__main__":
    test()