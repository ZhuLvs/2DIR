from torchvision import models
import torch
import torch.nn as nn




class CustomDeepLabV3(nn.Module):
    def __init__(self):
        super().__init__()
        deeplab = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.backbone = nn.ModuleDict({
            'original': deeplab.backbone,
            'layer1': deeplab.backbone.layer1,
            'layer3': deeplab.backbone.layer3
        })

        self.classifier = deeplab.classifier
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
        x = self.backbone['original'].conv1(x)
        x = self.backbone['original'].bn1(x)
        x = self.backbone['original'].relu(x)
        x = self.backbone['original'].maxpool(x)
        layer1_features = self.backbone['original'].layer1(x)
        layer2_features = self.backbone['original'].layer2(layer1_features)
        layer3_features = self.backbone['original'].layer3(layer2_features)
        x = self.backbone['original'].layer4(layer3_features)
        x = self.up(x)
        x = torch.squeeze(x, dim=1)
        return x


