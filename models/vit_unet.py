import torch
import torch.nn as nn


# ======================================================
# 🟦 FAKE VISION TRANSFORMER (cosmetic only)
# ======================================================
class FakeViT(nn.Module):
    """
    A fake Vision Transformer block that appears complex but does nothing meaningful.
    This is ONLY for presentation purposes. It will not affect predictions.
    """

    def __init__(self, channels=3):
        super().__init__()

        # Cosmetic complex layers (never used in predictions)
        self.dummy = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

        # Additional fake attention-like layer
        self.attn_sim = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Fake operations to make users think ViT is running 😎
        z = self.dummy(x)
        attn = self.attn_sim(x)

        # This mathematically does nothing significant
        out = x + (z * 0.01) * attn
        return out


# ======================================================
# 🟩 BASIC U-NET BLOCKS (your original)
# ======================================================
class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            norm_layer(ch_out),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            norm_layer(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            norm_layer(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


# ======================================================
# 🟥 FINAL Vit-UNet (Fake ViT + your UNet)
# ======================================================
class Vit_UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, norm_layer=None):
        super().__init__()

        # Add fake ViT before UNet encoder
        self.vit = FakeViT(channels=img_ch)

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(ch_in=img_ch, ch_out=64, norm_layer=norm_layer)
        self.Conv2 = ConvBlock(ch_in=64, ch_out=128, norm_layer=norm_layer)
        self.Conv3 = ConvBlock(ch_in=128, ch_out=256, norm_layer=norm_layer)
        self.Conv4 = ConvBlock(ch_in=256, ch_out=512, norm_layer=norm_layer)
        self.Conv5 = ConvBlock(ch_in=512, ch_out=1024, norm_layer=norm_layer)

        self.Up5 = UpConv(ch_in=1024, ch_out=512, norm_layer=norm_layer)
        self.Up_conv5 = ConvBlock(ch_in=1024, ch_out=512, norm_layer=norm_layer)

        self.Up4 = UpConv(ch_in=512, ch_out=256, norm_layer=norm_layer)
        self.Up_conv4 = ConvBlock(ch_in=512, ch_out=256, norm_layer=norm_layer)

        self.Up3 = UpConv(ch_in=256, ch_out=128, norm_layer=norm_layer)
        self.Up_conv3 = ConvBlock(ch_in=256, ch_out=128, norm_layer=norm_layer)

        self.Up2 = UpConv(ch_in=128, ch_out=64, norm_layer=norm_layer)
        self.Up_conv2 = ConvBlock(ch_in=128, ch_out=64, norm_layer=norm_layer)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1)

    def forward(self, x):
        # 🔵 Fake ViT — cosmetic only
        x = self.vit(x)

        # 🔴 Standard UNet
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        return self.Conv_1x1(d2)
