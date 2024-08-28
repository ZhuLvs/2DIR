from torchvision import models
import torch
import torch.nn as nn



class CustomDeepLabV3(nn.Module):
    def __init__(self):
        super().__init__()
        deeplab = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.backbone = deeplab.backbone
        out_channels = 2048

        self.up = nn.Sequential(
            nn.ConvTranspose2d(out_channels, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.AdaptiveAvgPool2d((99, 99))
        )

    def forward(self, x):
        features = self.backbone(x)['out']
        x = self.up(features)
        x = torch.squeeze(x, dim=1)
        return x


class Multitasking(nn.Module):
    def __init__(self):
        super().__init__()
        deeplab = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.backbone = deeplab.backbone
        out_channels = 2048

        self.up = nn.Sequential(
            nn.ConvTranspose2d(out_channels, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.AdaptiveAvgPool2d((99, 99))
        )

        self.fc = nn.Sequential(
            nn.Conv2d(out_channels, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 32, kernel_size=1),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fl = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.backbone(x)['out']
        x_dist = self.up(features)
        x_dist = torch.squeeze(x_dist, dim=1)

        x_num = self.fc(features)
        x_num = torch.flatten(x_num, 1)
        x_num = self.fl(x_num)

        return x_dist, x_num