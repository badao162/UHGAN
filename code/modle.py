
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import cv2
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import logging
import datetime
import sys
from pathlib import Path
import itertools
import pandas as pd
class CrossAttentionTransformer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, ff_hidden_dim=2048):
        super().__init__()

        # 确保 embed_dim 和 num_heads 是整数
        self.attn1 = nn.MultiheadAttention(embed_dim=int(embed_dim), num_heads=int(num_heads))  # 用于原始图像的注意力
        self.attn2 = nn.MultiheadAttention(embed_dim=int(embed_dim), num_heads=int(num_heads))  # 用于生成的掩膜的注意力
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, sat_imgs, fake_masks):
        # 将原始图像和生成的掩膜都通过注意力机制进行处理
        # 首先将两个输入进行拼接以便共享信息
        combined_input = torch.cat((sat_imgs, fake_masks), dim=1)  # (B, 2, H, W)

        # 进行空间扁平化以便输入到 Transformer 中
        B, C, H, W = combined_input.shape
        combined_input = combined_input.view(B, C, -1).permute(2, 0, 1)  # (H*W, B, C)

        # 对拼接后的输入应用自注意力机制
        x_ = self.norm1(combined_input)
        attn_output1, _ = self.attn1(x_, x_, x_)
        attn_output2, _ = self.attn2(x_, x_, x_)

        # 将两个注意力输出进行相加
        x = attn_output1 + attn_output2

        # 使用前馈神经网络处理注意力输出
        x_ = self.norm2(x)
        ff_output = self.ffn(x_)

        # 最终返回
        x = x + ff_output
        return x


class RoadDataset(Dataset):
    def __init__(self, root_dir, augment=True):
        self.sat_images = sorted(glob.glob(os.path.join(root_dir, "*_sat.jpg")))
        self.mask_images = sorted(glob.glob(os.path.join(root_dir, "*_mask.png")))
        self.augment = augment

        # 定义 RGB 图像的数据增强（适用于卫星图像）
        self.sat_transform = transforms.Compose([
            transforms.RandomResizedCrop(1024, scale=(0.8, 1.0)),  # 随机裁剪
            transforms.RandomHorizontalFlip(),  # 水平翻转
            transforms.RandomVerticalFlip(),  # 垂直翻转
            transforms.RandomRotation(30, interpolation=transforms.InterpolationMode.BILINEAR),  # 旋转（双线性插值）
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
            transforms.ToTensor(),
            transforms.Normalize([0.0166, 0.0219, 0.0190],[0.0077, 0.0060, 0.0056])

        ])

        # 定义掩码的变换（不包含颜色增强）
        self.mask_transform = transforms.Compose([
            transforms.RandomResizedCrop(1024, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30, interpolation=transforms.InterpolationMode.NEAREST),  # 旋转（最近邻插值）
            transforms.ToTensor()  # 直接转换为 Tensor
        ])

        # 基础转换（不含数据增强）
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.sat_images)

    def __getitem__(self, idx):
        # 读取原始图像
        sat_img = cv2.imread(self.sat_images[idx])
        sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)
        mask_img = cv2.imread(self.mask_images[idx], cv2.IMREAD_GRAYSCALE)

        # 转换为 PIL 格式
        sat_pil = Image.fromarray(sat_img)
        mask_pil = Image.fromarray(mask_img)

        # 进行数据增强（保持 mask 和 sat 的一致性）
        if self.augment:
            seed = torch.randint(0, 2**32, (1,)).item()  # 生成随机种子
            torch.manual_seed(seed)  # 让 sat_pil 的变换可复现
            sat_pil = self.sat_transform(sat_pil)
            torch.manual_seed(seed)  # 让 mask_pil 采用相同变换
            mask_pil = self.mask_transform(mask_pil)
        else:
            sat_pil = self.base_transform(sat_pil)
            mask_pil = transforms.ToTensor()(mask_pil)

        return sat_pil, mask_pil

# 生成器（U-Net）模型
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, ff_hidden_dim=2048):
        super().__init__()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Self-attention
        x_ = self.norm1(x)
        attn_output, _ = self.attn(x_, x_, x_)
        x = x + attn_output

        # Feed-forward network
        x_ = self.norm2(x)
        ff_output = self.ffn(x_)
        x = x + ff_output

        return x
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers += [
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
        self.convs = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.convs(x)
        skip = x
        x = self.pool(x)
        return x, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2, use_transpose=True):
        super().__init__()
        # 上采样模块
        if use_transpose:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1)
            )

        # 卷积模块
        layers = []
        merged_channels = in_channels + out_channels  # 跳跃连接拼接后的通道数
        for _ in range(num_convs):
            layers += [
                nn.Conv2d(merged_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            merged_channels = out_channels
        self.convs = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.convs(x)


    

class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器
        self.enc1 = EncoderBlock(3, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        
        # 添加通道注意力和空间注意力
        self.ca1 = ChannelAttention(64)
        self.sa1 = SpatialAttention()
        
        self.ca2 = ChannelAttention(128)
        self.sa2 = SpatialAttention()
        
        self.ca3 = ChannelAttention(256)
        self.sa3 = SpatialAttention()
        
        self.ca4 = ChannelAttention(512)
        self.sa4 = SpatialAttention()
        
        # 瓶颈层
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # 转换为512维度进行Transformer处理
        self.reduce_dim = nn.Conv2d(1024, 512, 1)
        
        # Transformer块用于捕获全局依赖
        self.transformer = TransformerBlock(embed_dim=512, num_heads=8)
        
        # 恢复维度
        self.restore_dim = nn.Conv2d(512, 1024, 1)

        # 解码器
        self.dec1 = DecoderBlock(1024, 512, use_transpose=True)
        self.dec2 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec4 = DecoderBlock(128, 64)
        
        # 添加细节增强分支
        self.detail_branch = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 输出层 - 融合细节分支
        self.final_conv = nn.Sequential(
            nn.Conv2d(64+32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        input_img = x  # 保存原始输入用于细节分支
        
        # 编码器 + 注意力
        x, s1 = self.enc1(x)  # 1024->512
        x = self.ca1(x)       # 通道注意力
        x = self.sa1(x)       # 空间注意力
        
        x, s2 = self.enc2(x)  # 512->256
        x = self.ca2(x)
        x = self.sa2(x)
        
        x, s3 = self.enc3(x)  # 256->128
        x = self.ca3(x)
        x = self.sa3(x)
        
        x, s4 = self.enc4(x)  # 128->64
        x = self.ca4(x)
        x = self.sa4(x)
        
        # 瓶颈
        x = self.enc5(x)  # 保持64x64
        
        # # 使用Transformer进行全局特征处理
        # batch_size, c, h, w = x.shape
        
        # # 降维以匹配transformer的嵌入维度
        # x_reduced = self.reduce_dim(x)
        
        # # 将特征展平为序列并应用Transformer
        # x_flat = x_reduced.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        # x_transformed = self.transformer(x_flat)
        
        # # 重塑回空间维度
        # x_spatial = x_transformed.permute(1, 2, 0).view(batch_size, 512, h, w)
        
        # # 恢复原始通道数
        # x = self.restore_dim(x_spatial)
        
        # 解码器
        x = self.dec1(x, s4)  # 64->128
        x = self.dec2(x, s3)  # 128->256
        x = self.dec3(x, s2)  # 256->512
        x = self.dec4(x, s1)  # 512->1024
        
        # 处理细节分支
        details = self.detail_branch(input_img)
        # 确保尺寸一致
        if details.size()[2:] != x.size()[2:]:
            details = F.interpolate(details, size=x.size()[2:], mode='bilinear', align_corners=True)
            
        # 融合主干特征和细节特征
        combined = torch.cat([x, details], dim=1)
        
        # 最终预测
        return self.final_conv(combined)

class UNetGenerator1(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        def conv_block(in_channels, out_channels, bn=True):
            layers = [
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            if bn: layers.append(nn.BatchNorm2d(out_channels))
            return nn.Sequential(*layers)

        # 下采样
        self.down1 = conv_block(3, 32, bn=False)
        self.down2 = conv_block(32, 64)
        self.down3 = conv_block(64, 128)
        self.down4 = conv_block(128, 256)
        self.down5 = conv_block(256, 512)

        # Transformer模块
        self.transformer = TransformerBlock(embed_dim=512, num_heads=8)  # 修改为embed_dim=512

        # 上采样
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True))
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True))
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 1, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True))
        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True))
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True))
        self.up9 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 6, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True))
        self.up10 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True))
        self.final = nn.Sequential(
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Tanh())  # 输出一个0-1范围的图像

    def forward(self, x):
        # 下采样
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        # 使用Transformer进行特征增强
        transformer_out = self.transformer(d5.flatten(2).permute(2, 0, 1))
        transformer_out = transformer_out.permute(1, 2, 0).view_as(d5)

        # 使用上采样将尺寸从 32x32 调整为 64x64
        transformer_out = torch.nn.functional.interpolate(transformer_out, size=(64, 64), mode='bilinear',
                                                          align_corners=True)

        # 上采样+跳跃连接
        #print(transformer_out.shape)
        u1 = self.up1(transformer_out)# 1->256,
        # print(f"u1 shape: {u1.shape}, d4 shape: {d4.shape}")

        u1 = torch.cat([u1, d4], 1)#256+256,
        #print(f"u1 shape: {u1.shape}")
        u2 = self.up2(u1)#512->128
        #print(f"u2 shape: {u2.shape}, d3 shape: {d3.shape}")
        u2 = torch.cat([u2, d3], 1)  # 128+128,
        u3=self.up3(u2)#*2,256
        u4 = self.up4(u3)  # *2,256->128
        u5 = self.up5(u4)  # *4,128->128
        u6 = self.up6(u5)  # *4,128->64
        #print(f"u6 shape: {u6.shape}, d2 shape: {d2.shape}")
        u6=torch.cat([u6, d2], 1)#*2 64+64,
        u6=self.up6(u6)#*4,128->64
        u7=self.up7(u6)#*8,64->64
        u8=self.up8(u7)#*8,64->32
        u9=self.up9(u8)#**16.32->32
        #print(f"u9 shape: {u9.shape}, d1 shape: {d1.shape}")
        u10=torch.cat([u9,d1],1)#**16.32+32
        #print(u10.shape)
        u10=self.up8(u10)#**16.64->32
        u10=self.up10(u10)

        #print((u10.shape))
        #print(self.final(u10).shape)
        return self.final(u10)
    


class UNetGenerator_union(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器
        self.enc1 = EncoderBlock(1, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # 解码器
        self.dec1 = DecoderBlock(1024, 512, use_transpose=True)
        self.dec2 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec4 = DecoderBlock(128, 64)
        self.transformer = TransformerBlock(embed_dim=512, num_heads=8)
        # 输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 编码器
        x, s1 = self.enc1(x)  # 1024->512
        x, s2 = self.enc2(x)  # 512->256
        x, s3 = self.enc3(x)  # 256->128
        x, s4 = self.enc4(x)  # 128->64
        x = self.enc5(x)  # 保持64x64
                
        # # 使用Transformer进行全局特征处理
        # # 首先将特征展平为序列
        # batch_size, c, h, w = x.shape
        
        # # 修改这里：确保展平后的特征维度与Transformer期望的嵌入维度匹配
        # if c != 512:  # 假设Transformer的embed_dim=512
        #     # 使用1x1卷积调整通道数
        #     channel_adjust = nn.Conv2d(c, 512, kernel_size=1).to(x.device)
        #     x = channel_adjust(x)
        #     c = 512
        
        # x_flat = x.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        
        # # 应用Transformer
        # x_transformed = self.transformer(x_flat)
        
        # # 将特征重新塑造为原始形状
        # x = x_transformed.permute(1, 2, 0).view(batch_size, c, h, w)
        
        # # 如果之前调整了通道数，现在需要调整回来
        # if c != 1024:
        #     channel_restore = nn.Conv2d(c, 1024, kernel_size=1).to(x.device)
        #     x = channel_restore(x)
        
        #解码器
        x = self.dec1(x, s4)  # 64->128
        x = self.dec2(x, s3)  # 128->256
        x = self.dec3(x, s2)  # 256->512
        x = self.dec4(x, s1)  # 512->1024

        return self.final_conv(x)

# 判别器（PatchGAN）模型
class Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, img, mask):



        #print(f"img shape: {img.shape}, mask shape: {mask.shape}")
        x = torch.cat([img, mask], 1)
        #print(f"x:{x.shape}")
        return self.model(x)

    class Discriminator(nn.Module):
        def __init__(self, in_channels=4):
            super().__init__()

            self.model = nn.Sequential(
                nn.Conv2d(in_channels, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(128, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(256, 512, 4, 1, 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(512, 1, 4, 1, 1),
                nn.Sigmoid()
            )

        def forward(self, img, mask):
            # print(f"img shape: {img.shape}, mask shape: {mask.shape}")
            x = torch.cat([img, mask], 1)
            # print(f"x:{x.shape}")
            return self.model(x)