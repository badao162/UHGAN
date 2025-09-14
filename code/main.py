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
import modle
# 配置参数
config = {
    "data_path": "/home/my/gbznt/SHUJU_train",  # 数据路径
    "test_data_path": "/home/my/gbznt/SHUJU_test",
    "batch_size": 1,  # 批量大小
    "lr": 0.0002,  # 学习率
    "epochs": 120,  # 训练轮数
    "img_size": 1024,  # 图片尺寸
    "channels": 3,  # 通道数
    "device": "cuda:1" if torch.cuda.is_available() else "cpu",  # 设备
    "output_dir": "/home/my/gbznt/union_photo",  # 输出路径
    "test_output":"/home/my/gbznt/test_results",
    "lambda_HT": 10,  # Hough变换损失的权重
    "lambda_geo": 10  # 几何损失的权重
}
param_grid = {
    'lambda_adv': [1.0, 0.5],
    'lambda_seg': [0.1, 1.0, 0.5],
    'lambda_ht': [0.1, 1.0, 0.5]
}
# 创建所有可能的参数组合
grid_search_params = list(itertools.product(
    param_grid['lambda_adv'],
    param_grid['lambda_seg'],
    param_grid['lambda_ht']
))

# 用于存储所有网格搜索的结果
grid_search_results = []

# 配置日志记录
def setup_logger(name, log_dir='/home/my/logs'):
    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 当前时间作为日志文件名的一部分
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/{name}_{timestamp}.log"
    
    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # 设置logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 创建主日志记录器
logger = setup_logger('road_extraction')
logger.info("初始化训练配置和模型")

# 将配置参数记录到日志
import torch
import torch.nn as nn


class Discriminator_union(nn.Module):
    def __init__(self, in_channels=2):
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
def calculate_metrics(pred, target):
    pred = pred.flatten()
    target = target.flatten()
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)


    # print("Unique values in target:", np.unique(target))
    # print("Unique values in pred:", np.unique(pred))
    tp = np.sum((pred == 1) & (target == 1))
    fp = np.sum((pred == 1) & (target == 0))
    fn = np.sum((pred == 0) & (target == 1))
    # print(tp)
    # print(fp)
    # print(fn)


    precision = tp / (tp + fp + 1e-10)  # 防止除以0
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return precision, recall, f1
# 记录配置信息
for key, value in config.items():
    logger.info(f"配置参数 {key}: {value}")



# 可视化对比函数
def plot_comparison(sat_img, real_mask, pred_mask,union_pred_mask, save_path):
    plt.figure(figsize=(15, 5))

    # 反归一化卫星图像
    sat_img = sat_img.permute(1, 2, 0).numpy() * 0.5 + 0.5

    # 绘制卫星图像
    plt.subplot(1, 4, 1)
    plt.imshow(sat_img)
    plt.title("Satellite Image")
    plt.axis('off')

    # 真实道路
    plt.subplot(1, 4, 2)
    plt.imshow(real_mask.squeeze(), cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')

    # 预测结果1
    plt.subplot(1, 4, 3)
    plt.imshow(pred_mask.squeeze(), cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')
    # 预测结果2
    plt.subplot(1, 4, 4)
    plt.imshow(union_pred_mask.squeeze(), cmap='gray')
    plt.title("Predicted union_Mask")
    plt.axis('off')

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import glob
import os

# 配置参数


# 生成器结构（包含编码器-解码器和残差块）
class PLGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        # 残差块
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(6)]
        )

        # 解码器（生成高亮图像）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

        # 语义分割解码器
        self.semantic_decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 编码器提取特征
        features = self.encoder(x)
        features = self.res_blocks(features)

        # 生成高亮图像
        highlighted = self.decoder(features)

        # 语义分割
        segmentation = self.semantic_decoder(features)

        return highlighted, segmentation


# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, 1, 1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(),
#             nn.Conv2d(channels, channels, 3, 1, 1),
#             nn.BatchNorm2d(channels)
#         )

#     def forward(self, x):
#         return x + self.block(x)


# 判别器结构（Markovian判别器）
class PLDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)




class DifferentiableHoughTransform(torch.autograd.Function):
    """
    一个实现了直通估计器 (STE) 的可微分霍夫变换模块。
    """
    @staticmethod
    def hough_transform(image, theta_bins, cos_theta, sin_theta):
        """
        高效的、可批处理的霍夫变换实现。
        """
        if image.dim() != 4 or image.size(1) != 1:
            raise ValueError("输入图像必须是 (B, 1, H, W) 的4D张量")
        
        B, _, H, W = image.shape
        device = image.device
        
        # 计算 rho 的最大值和离散化的 bin 数量
        rho_max = torch.sqrt(torch.tensor(H**2 + W**2, dtype=torch.float32, device=device))
        n_rho = int(torch.ceil(rho_max)) * 2  # 从 -rho_max 到 +rho_max
        
        # 创建累加器
        accumulator = torch.zeros(B, theta_bins, n_rho, device=device)
        
        # 找到所有活动像素的坐标 (y, x)
        # nonzero() 返回一个元组，每个维度一个张量
        #y_coords, x_coords = torch.nonzero(image.squeeze(1) > 0.5, as_tuple=True)
        # 获取每个活动像素对应的批次索引
        batch_indices, y_coords, x_coords = torch.nonzero(image.squeeze(1) > 0.5, as_tuple=True)
       

        if x_coords.numel() == 0:
            return accumulator, None # 如果没有活动像素，直接返回

        # 扩展坐标和角度以进行批处理计算
        x_coords_exp = x_coords.float().view(-1, 1)
        y_coords_exp = y_coords.float().view(-1, 1)
        cos_theta_exp = cos_theta.view(1, -1)
        sin_theta_exp = sin_theta.view(1, -1)
        
        # 计算所有活动像素在所有角度下的 rho 值
        rho_values = x_coords_exp * cos_theta_exp + y_coords_exp * sin_theta_exp
        
        # 将 rho 值映射到累加器的索引
        # 加上 n_rho // 2 是为了将范围从 [-rho_max, rho_max] 移动到 [0, 2*rho_max]
        rho_indices = (rho_values + n_rho // 2).long()
        
        # 准备 scatter_add_ 的索引
        theta_indices = torch.arange(theta_bins, device=device).view(1, -1).expand(x_coords.size(0), -1)
        batch_indices_exp = batch_indices.view(-1, 1).expand(-1, theta_bins)
        
        # 使用 scatter_add_ 进行高效投票
        # .view(-1) 将所有索引和值展平为一维
        accumulator.view(-1).scatter_add_(
            0,
            (batch_indices_exp.flatten() * theta_bins * n_rho + 
             theta_indices.flatten() * n_rho + 
             rho_indices.flatten()),
            torch.ones_like(rho_indices, dtype=torch.float32).flatten()
        )
        
        # 保存反向传播所需的信息
        ctx_info = (batch_indices, x_coords, y_coords, image.shape)
        return accumulator, ctx_info

    @staticmethod
    def forward(ctx, pred_probs, target_mask, theta_bins, cos_theta, sin_theta):
        """
        前向传播：计算霍夫变换并返回L1损失。
        """
        # 1. 二值化预测（这是不可微步骤）
        pred_mask = (pred_probs > 0.5).float()
        
        # 2. 对预测和目标进行霍夫变换
        result_pred = DifferentiableHoughTransform.hough_transform(pred_mask, theta_bins, cos_theta, sin_theta)
        if isinstance(result_pred, tuple):
            accum_pred, ctx_info_pred = result_pred
        else:
            accum_pred = result_pred
            ctx_info_pred = None

        accum_target, _ = DifferentiableHoughTransform.hough_transform(target_mask, theta_bins, cos_theta, sin_theta)
        
        
        # 3. 计算L1损失
        loss = torch.abs(accum_pred - accum_target).mean()
        
        # 4. 保存反向传播所需的信息
        # 我们需要累加器的差值符号（L1损失的梯度）和预测掩码的上下文信息
        ctx.save_for_backward(accum_pred, accum_target)
        ctx.ctx_info_pred = ctx_info_pred
        ctx.pred_probs_shape = pred_probs.shape
        
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：使用STE近似梯度。
        """
        accum_pred, accum_target = ctx.saved_tensors
        ctx_info_pred = ctx.ctx_info_pred
        
        # 如果前向传播时没有活动像素，则梯度为零
        if ctx_info_pred is None:
            return torch.zeros(ctx.pred_probs_shape, device=accum_pred.device), None, None, None, None

        batch_indices, x_coords, y_coords, image_shape = ctx_info_pred
        
        # 1. 计算损失对累加器的梯度
        # L1损失的梯度是差值的符号
        grad_accum = grad_output * torch.sign(accum_pred - accum_target) / accum_pred.numel()
        
        # 2. 反向霍夫变换：将梯度从累加器空间“散播”回图像空间
        # 这是 ∂L/∂M 的计算
        grad_mask = torch.zeros(image_shape, device=grad_accum.device)
        
        # 从累加器中收集对应于每个投票的梯度值
        # (这部分比较复杂，一个简化的、但效果类似的近似是，
        # 将每个活动像素的梯度视为其所有投票的梯度之和的平均)
        # 为了高效，我们直接将梯度赋给活动像素
        # 这是一个有效的近似，避免了复杂的反向散射
        grad_per_pixel = grad_accum.mean() # 使用梯度的平均值作为每个像素的梯度
        grad_mask[batch_indices, 0, y_coords, x_coords] = grad_per_pixel

        # 3. STE: 将 ∂L/∂M 直接作为 ∂L/∂P
        # grad_pred_probs ≈ grad_mask
        grad_pred_probs = grad_mask
        
        # 返回的梯度数量必须与forward的输入参数数量一致
        return grad_pred_probs, None, None, None, None

class HoughLoss(nn.Module):
    """
    可微分霍夫变换损失。
    
    该损失函数将预测的概率图和真实掩码图转换为霍夫空间，
    然后计算它们在霍夫空间中的L1距离。
    通过使用直通估计器（STE），该过程在反向传播中是可微的。
    """
    def __init__(self, theta_bins=180):
        super().__init__()
        if theta_bins <= 0:
            raise ValueError("theta_bins 必须是正整数")
        self.theta_bins = theta_bins
        
        # 预计算 sin 和 cos 值以提高效率
        thetas = torch.linspace(0, torch.pi, theta_bins, dtype=torch.float32)
        self.register_buffer("cos_theta", torch.cos(thetas))
        self.register_buffer("sin_theta", torch.sin(thetas))

    def forward(self, pred_probs, target_mask):
        """
        计算霍夫损失。
        
        参数:
            pred_probs (torch.Tensor): 预测的概率图，形状为 (B, 1, H, W)，值在 [0, 1] 之间。
            target_mask (torch.Tensor): 真实的二值掩码，形状为 (B, 1, H, W)，值为 0 或 1。
        
        返回:
            torch.Tensor: 一个标量，表示计算出的霍夫损失。
        """
        if pred_probs.dim() != 4 or target_mask.dim() != 4:
            raise ValueError("输入张量必须是4D (B, C, H, W)")
        if pred_probs.size(1) != 1 or target_mask.size(1) != 1:
            raise ValueError("输入通道数必须为1")
        if pred_probs.shape != target_mask.shape:
            raise ValueError("输入和目标的形状必须匹配")

        return DifferentiableHoughTransform.apply(pred_probs, target_mask, self.theta_bins, self.cos_theta, self.sin_theta)


class GeometricTransform:
    def __init__(self):
        self.theta = torch.tensor([30])  # 示例旋转角度

    def __call__(self, x):
        # 应用旋转
        return torch.rot90(x, 1, [2, 3])

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0):
        """
        alpha小beta大会更加关注召回率提升
        """
        super().__init__()
        self.alpha = alpha  # 控制精确率权重
        self.beta = beta   # 控制召回率权重 (设置更大以提高召回率)
        self.gamma = gamma  # 聚焦参数
        
    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        
        # 平滑处理
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (y_pred * y_true).sum()
        FP = ((1-y_true) * y_pred).sum()
        FN = (y_true * (1-y_pred)).sum()
        
        Tversky = (TP + 1.0) / (TP + self.alpha*FP + self.beta*FN + 1.0)
        FocalTversky = (1 - Tversky)**self.gamma
        
        return FocalTversky
# 初始化模型
hough_loss = HoughLoss()
hough_loss = HoughLoss().to(config["device"]) 
geo_transform = GeometricTransform()

generator = UNetGenerator().to(config["device"])
discriminator = Discriminator().to(config["device"])
generator_union = UNetGenerator_union().to(config["device"])
#discriminator_union = Discriminator_union().to(config["device"])

# 定义损失函数和优化器
#criterion_GAN = nn.MSELoss()  # 用最小二乘法替代BCELoss
criterion_GAN= nn.BCELoss()
criterion_pixel = nn.L1Loss()
# 判别器损失

#criterion_GAN_union = nn.MSELoss()  # 用最小二乘法替代BCELoss
criterion_GAN_union = nn.BCELoss()


criterion_pixel_union = nn.L1Loss()

optimizer_G = optim.Adam(generator.parameters(), lr=0.0013, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.5, 0.999))
optimizer_G_union = optim.Adam(generator_union.parameters(), lr=0.00013, betas=(0.5, 0.999))
#optimizer_D_union = optim.Adam(discriminator_union.parameters(), lr=0.00013, betas=(0.5, 0.999))

# 学习率调度器（模拟退火）
scheduler_G = CosineAnnealingLR(optimizer_G, T_max=config["epochs"], eta_min=0)
scheduler_D = CosineAnnealingLR(optimizer_D, T_max=config["epochs"], eta_min=0)
scheduler_G_union = CosineAnnealingLR(optimizer_G_union, T_max=config["epochs"], eta_min=0)
#scheduler_D_union = CosineAnnealingLR(optimizer_D_union, T_max=config["epochs"], eta_min=0)

# 数据加载
dataset = RoadDataset(config["data_path"])
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
# 训练循环
import matplotlib.pyplot as plt
import os
"""
torch.save({
    'epoch': epoch,  # 保存当前epoch
    'generator_state_dict': generator.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict(),
    'scheduler_G_state_dict': scheduler_G.state_dict(),
    'scheduler_D_state_dict': scheduler_D.state_dict(),
    'lr_G': optimizer_G.param_groups[0]['lr'],  # 保存学习率
    'lr_D': optimizer_D.param_groups[0]['lr'],  # 保存学习率
    'generator_union_state_dict': generator_union.state_dict(),
    'optimizer_union_G_state_dict': optimizer_G_union.state_dict(),
    'discriminator_union_state_dict': discriminator_union.state_dict(),
    'optimizer_union_D_state_dict': optimizer_D_union.state_dict(),
    'scheduler_union_G_state_dict': scheduler_G_union.state_dict(),
    'scheduler_union_D_state_dict': scheduler_D_union.state_dict(),
    'lr_G_union': optimizer_G_union.param_groups[0]['lr'],  # 保存学习率
    'lr_D_union': optimizer_D_union.param_groups[0]['lr'],  # 保存学习率
}, f"/home/my/gbznt/union_model/checkpoint_final.pth")
"""
start_epoch=0

# checkpoint = torch.load("/home/my/gbznt/union_model/checkpoint_epoch_50.pth")
# for param_group in optimizer_G_union.param_groups:
#     param_group['lr'] = checkpoint['lr_G']

# # for param_group in optimizer_D_union.param_groups:
# #     param_group['lr'] = checkpoint['lr_D']

# for param_group in optimizer_G.param_groups:
#     param_group['lr'] = checkpoint['lr_G_union']

# # for param_group in optimizer_D.param_groups:
# #     param_group['lr'] = checkpoint['lr_D_union']

# # 恢复模型的状态字典
# generator.load_state_dict(checkpoint['generator_state_dict'])
# discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
# generator_union.load_state_dict(checkpoint['generator_union_state_dict'])
# #discriminator_union.load_state_dict(checkpoint['discriminator_union_state_dict'])

# # 恢复优化器的状态字典
# optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
# optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
# optimizer_G_union.load_state_dict(checkpoint['optimizer_union_G_state_dict'])
# #optimizer_D_union.load_state_dict(checkpoint['optimizer_union_D_state_dict'])

# # 恢复学习率调度器的状态字典
# scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
# scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
# scheduler_G_union.load_state_dict(checkpoint['scheduler_union_G_state_dict'])
# #scheduler_D_union.load_state_dict(checkpoint['scheduler_union_D_state_dict'])
# # 恢复训练的epoch
# start_epoch = checkpoint['epoch']  # 继续训练时，起始epoch为保存时的epoch+1



print(f"Restoring model and optimizer states, continuing from epoch {start_epoch}.")
# 修改训练循环，添加日志记录
# 训练循环
d_losses = []  # 用于存储判别器的损失
g_losses = []  # 用于存储生成器的损失
d_losses_ave = 0  # 用于存储判别器的损失
g_losses_ave = 0  # 用于存储生成器的损失
d_losses_fully = []  # 用于存储判别器的损失
g_losses_fully = []

# 获取模型参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 记录模型结构和参数数量
logger.info(f"Generator 参数数量: {count_parameters(generator):,}")
logger.info(f"Discriminator 参数数量: {count_parameters(discriminator):,}")
logger.info(f"Generator Union 参数数量: {count_parameters(generator_union):,}")
#logger.info(f"Discriminator Union 参数数量: {count_parameters(discriminator_union):,}")

# 如果是从检查点恢复训练，记录相关信息
if start_epoch > 0:
    logger.info(f"从 Epoch {start_epoch} 恢复训练")

# 记录数据集信息
logger.info(f"训练数据集大小: {len(dataloader)}")

# for epoch in range(start_epoch+1, config["epochs"]+1):
#     logger.info(f"开始 Epoch {epoch}/{config['epochs']}")
#     epoch_start_time = datetime.datetime.now()
    
#     d_losses = []  # 用于存储判别器的损失
#     total_loss_list = []  # 用于存储生成器的损失
#     d_losses_ave = 0  # 用于存储判别器的损失
#     g_losses_ave = 0  # 用于存储生成器的损失
    
#     # 训练批次循环
#     for i, (sat_imgs, real_masks) in enumerate(dataloader):
#         batch_start_time = datetime.datetime.now()
        
#         # 数据迁移到设备
#         sat_imgs = sat_imgs.to(config["device"])
#         real_masks = real_masks.to(config["device"])

#         # 生成假mask
#         fake_masks = generator(sat_imgs)

#         # 训练判别器
#         optimizer_D.zero_grad()
#         # 真实数据损失
#         real_validity = discriminator(sat_imgs, real_masks)
#         real_loss = criterion_GAN(real_validity, torch.ones_like(real_validity))
#         # 假数据损失
#         fake_validity = discriminator(sat_imgs, fake_masks.detach())
#         fake_loss = criterion_GAN(fake_validity, torch.zeros_like(fake_validity))
#         d_loss = (real_loss + fake_loss) / 2
#         d_loss.backward()
#         optimizer_D.step()

#         # 训练生成器
#         optimizer_G.zero_grad()
#         # GAN损失 + L1像素损失
#         fake_validity = discriminator(sat_imgs, fake_masks)
#         adv_loss = criterion_GAN(fake_validity, torch.ones_like(fake_validity))
#         # 分割损失
#         seg_loss = nn.BCEWithLogitsLoss()(fake_masks, real_masks)
#         # Hough变换损失
#         ht_loss = hough_loss(fake_masks, real_masks) 

#         # 几何一致性损失
#         transformed = geo_transform(sat_imgs)
#         seg_trans = generator(transformed)
#         geo_loss = torch.mean(torch.abs(fake_masks - geo_transform(seg_trans)))
        
#         total_loss = adv_loss +0.00000000000001*geo_loss + seg_loss + ht_loss

#         total_loss.backward()
#         optimizer_G.step()

#         # 在第二次训练时使用纯UNet（不使用GAN）
#         fake_masks_detached = fake_masks.detach()  # 切断与原始生成器的计算图
        
#         # 训练第二层UNet（不使用对抗训练）
#         optimizer_G_union.zero_grad()
#         fake_masks_union = generator_union(fake_masks_detached)  # 输入已detach的第一层结果
        
#         # 只使用L1损失或MSE损失进行监督学习
#         g_loss_union = criterion_pixel_union(fake_masks_union, real_masks)
#         g_loss_union.backward()
#         optimizer_G_union.step()
for run_id, params in enumerate(grid_search_params):
    lambda_adv, lambda_seg, lambda_ht = params
    
    # --- 每个新搜索组合都需要重新初始化所有内容 ---
    
    # 为本次运行创建一个专用的日志记录器
    run_logger_name = f'road_extraction_run_{run_id}'
    logger = setup_logger(run_logger_name)
    
    logger.info("="*50)
    logger.info(f"开始网格搜索运行: {run_id + 1}/{len(grid_search_params)}")
    logger.info(f"当前参数: lambda_adv={lambda_adv}, lambda_seg={lambda_seg}, lambda_ht={lambda_ht}")
    logger.info("="*50)

    # 初始化模型
    hough_loss = HoughLoss().to(config["device"]) 
    geo_transform = GeometricTransform()

    generator = UNetGenerator().to(config["device"])
    discriminator = Discriminator().to(config["device"])
    generator_union = UNetGenerator_union().to(config["device"])

    # 定义损失函数和优化器
    criterion_GAN = nn.BCELoss()
    criterion_pixel = nn.L1Loss()
    criterion_pixel_union = nn.L1Loss()

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0013, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.5, 0.999))
    optimizer_G_union = optim.Adam(generator_union.parameters(), lr=0.00013, betas=(0.5, 0.999))

    # 学习率调度器
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=config["epochs"], eta_min=0)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=config["epochs"], eta_min=0)
    scheduler_G_union = CosineAnnealingLR(optimizer_G_union, T_max=config["epochs"], eta_min=0)

    # 数据加载
    dataset = RoadDataset(config["data_path"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    start_epoch = 0
    best_f1_for_this_run = 0.0 # 记录当前运行的最佳F1

    # ... (打印模型参数数量的代码可以保留在这里) ...
    logger.info(f"Generator 参数数量: {sum(p.numel() for p in generator.parameters() if p.requires_grad):,}")
    # ... (其他模型参数打印) ...

    # --- 训练循环 ---
    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        logger.info(f"开始 Epoch {epoch}/{config['epochs']}")
        # ... (epoch 内的损失列表初始化等代码保持不变) ...
        epoch_start_time = datetime.datetime.now()
        # 训练批次循环
        for i, (sat_imgs, real_masks) in enumerate(dataloader):
            # ... (数据迁移和生成假mask的代码保持不变) ...
            sat_imgs = sat_imgs.to(config["device"])
            real_masks = real_masks.to(config["device"])
            fake_masks = generator(sat_imgs)
            batch_start_time = datetime.datetime.now()
            total_loss_list = []  # 用于存储生成器的损失
            # --- 训练判别器 (保持不变) ---
            optimizer_D.zero_grad()
            real_validity = discriminator(sat_imgs, real_masks)
            real_loss = criterion_GAN(real_validity, torch.ones_like(real_validity))
            fake_validity = discriminator(sat_imgs, fake_masks.detach())
            fake_loss = criterion_GAN(fake_validity, torch.zeros_like(fake_validity))
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # --- 训练生成器 (核心修改) ---
            optimizer_G.zero_grad()
            fake_validity = discriminator(sat_imgs, fake_masks)
            adv_loss = criterion_GAN(fake_validity, torch.ones_like(fake_validity))
            seg_loss = nn.BCEWithLogitsLoss()(fake_masks, real_masks)
            ht_loss = hough_loss(fake_masks, real_masks) 
            transformed = geo_transform(sat_imgs)
            seg_trans = generator(transformed)
            geo_loss = torch.mean(torch.abs(fake_masks - geo_transform(seg_trans)))
            
            # --- 使用网格搜索的权重动态计算总损失 ---
            total_loss = (lambda_adv * adv_loss + 
                          lambda_seg * seg_loss + 
                          lambda_ht * ht_loss + 
                          0.00000000000001 * geo_loss)
            # --- 使用网格搜索的权重动态计算总损失 ---
            # total_loss = (lambda_adv * adv_loss + 
            #               lambda_seg * seg_loss + 
            #               lambda_ht * ht_loss + 
            #               0.001 * geo_loss)
            total_loss.backward()
            optimizer_G.step()

            # --- 第二层UNet训练 (保持不变) ---
            fake_masks_detached = fake_masks.detach()
            optimizer_G_union.zero_grad()
            fake_masks_union = generator_union(fake_masks_detached)
            g_loss_union = criterion_pixel_union(fake_masks_union, real_masks)
            g_loss_union.backward()
            optimizer_G_union.step()        
    
        
            batch_end_time = datetime.datetime.now()
            batch_duration = (batch_end_time - batch_start_time).total_seconds()

        # 记录训练信息
        if i % 10 == 0 or i == len(dataloader) - 1:  # 每10个batch或最后一个batch记录一次
            logger.info(
                f"[Epoch {epoch}/{config['epochs']}] [Batch {i}/{len(dataloader)}] "
                f"[D loss: {d_loss.item():.4f}] [G loss: {total_loss.item():.4f}] "
                f"[D_union loss: {g_loss_union.item():.4f}] [G_union loss: {g_loss_union.item():.4f}] "
                f"[Batch time: {batch_duration:.2f}s]"
            )
            
            # 详细损失分解
            logger.debug(
                f"损失详情 - adv_loss: {seg_loss .item():.4f}, "#seg_loss: {seg_loss.item():.4f}, 
                #f"ht_loss: {ht_loss.item():.4f}, geo_loss: {geo_loss.item():.4f}"
            )
        
        d_losses.append(d_loss.item())
        total_loss_list.append(total_loss.item())
        g_losses_ave += total_loss.item()
        d_losses_ave += d_loss.item()
        
        # 每100个batch保存图片
        if i % 50 == 0:
            with torch.no_grad():
                fake = generator(sat_imgs[:1])
                fake2 = generator_union(fake_masks[:1])
                # 确保 fake 图像的通道数为 3（RGB）
                if fake.size(1) == 1:  # 如果是单通道图像，转换为3通道
                    fake = fake.repeat(1, 3, 1, 1)  # 重复通道，变成3通道图像
                if fake2.size(1) == 1:  # 如果是单通道图像，转换为3通道
                    fake2 = fake2.repeat(1, 3, 1, 1)  # 重复通道，变成3通道图像
                # 确保 real_masks 是三通道
                if real_masks.size(1) == 1:  # 如果是单通道图像，转换为3通道
                    real_masks = real_masks.repeat(1, 3, 1, 1)  # 重复通道，变成3通道图像

                # 使用matplotlib绘制图片，并标注图片类型
                plt.figure(figsize=(12, 6))

                # 第一张图：satellite image
                plt.subplot(1, 4, 1)
                plt.imshow(sat_imgs[0].cpu().permute(1, 2, 0) * 0.5 + 0.5)
                plt.title("Satellite Image")
                plt.axis('off')

                # 第二张图：generated mask
                plt.subplot(1, 4, 2)
                plt.imshow(fake2[0].cpu().permute(1, 2, 0) * 0.5 + 0.5)
                plt.title("Generated Mask")
                plt.axis('off')
                # 第三张图：generated mask2
                plt.subplot(1, 4, 3)
                plt.imshow(fake[0].cpu().permute(1, 2, 0) * 0.5 + 0.5)
                plt.title("Generated Mask2")
                plt.axis('off')
                # 第三张图：real mask
                plt.subplot(1, 4, 4)
                plt.imshow(real_masks[0].cpu().permute(1, 2, 0) * 0.5 + 0.5)
                plt.title("Real Mask")
                plt.axis('off')

                # 保存图像到指定目录
                if not os.path.exists(config["output_dir"]):
                    os.makedirs(config["output_dir"])

                plt.savefig(f"{config['output_dir']}/results_epoch{epoch}_batch{i}.png")
                plt.close()
        # 计算平均损失
        epoch_end_time = datetime.datetime.now()
        epoch_duration = (epoch_end_time - epoch_start_time).total_seconds() / 60.0  # 转换为分钟
        g_losses_ave /= len(dataloader)
        d_losses_ave /= len(dataloader)
        d_losses_fully.append(d_losses_ave)
        g_losses_fully.append(g_losses_ave)
        
        # 更新学习率
        scheduler_G.step()
        scheduler_D.step()
        scheduler_G_union.step()
        #scheduler_D_union.step()

        # 获取当前学习率
        lr_G = optimizer_G.param_groups[0]['lr']
        lr_D = optimizer_D.param_groups[0]['lr']
        lr_G_union = optimizer_G_union.param_groups[0]['lr']
        #lr_D_union = optimizer_D_union.param_groups[0]['lr']

        # 记录每轮的统计信息
        logger.info(f"Epoch {epoch}/{config['epochs']} 完成 - 用时: {epoch_duration:.2f}分钟")
        logger.info(f"平均损失 - Generator: {g_losses_ave:.4f}, Discriminator: {d_losses_ave:.4f}")
        logger.info(f"学习率 - Generator: {lr_G:.6f}, Discriminator: {lr_D:.6f}, "
                    f"Generator_union: {lr_G_union:.6f}, Discriminator_union: {lr_G_union:.6f}")

        # 绘制并保存损失曲线
        loss_plot_path = f"/home/my/gbznt/union_loss/loss_batch_{epoch}.png"
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(d_losses, label='Discriminator Loss')
            plt.plot(total_loss_list, label='Generator Loss')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.title(f'Epoch {epoch} - Discriminator and Generator Loss')
            plt.legend()
            plt.savefig(loss_plot_path)
            plt.close()
            logger.info(f"保存损失曲线到 {loss_plot_path}")
        except Exception as e:
            logger.error(f"保存损失曲线失败: {e}")

        # 保存检查点
        if epoch % 30 == 0:
            checkpoint_path = f"/home/my/gbznt/union_model/run{run_id}_checkpoint_epoch_{epoch}.pth"
            try:
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': generator.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                    'scheduler_G_state_dict': scheduler_G.state_dict(),
                    'scheduler_D_state_dict': scheduler_D.state_dict(),
                    'lr_G': lr_G,
                    'lr_D': lr_D,
                    'generator_union_state_dict': generator_union.state_dict(),
                    'optimizer_union_G_state_dict': optimizer_G_union.state_dict(),
                    #'discriminator_union_state_dict': discriminator_union.state_dict(),
                    #'optimizer_union_D_state_dict': optimizer_D_union.state_dict(),
                    'scheduler_union_G_state_dict': scheduler_G_union.state_dict(),
                    #'scheduler_union_D_state_dict': scheduler_D_union.state_dict(),
                    'lr_G_union': lr_G_union,
                    #'lr_D_union': lr_D_union,
                }, checkpoint_path)
                logger.info(f"保存检查点到 {checkpoint_path}")
            except Exception as e:
                logger.error(f"保存检查点失败: {e}")
                
            # 测试模型
            logger.info(f"开始使用 Epoch {epoch} 的检查点测试模型")
            try:
                metrics = test_model(epoch, checkpoint_path)
                # 记录测试指标
                for key, value in metrics.items():
                    logger.info(f"测试指标 {key}: {value:.4f}")
                # 我们以 union_f1 作为评判标准
                current_f1 = metrics.get("union_f1", 0.0)
                if current_f1 > best_f1_for_this_run:
                    best_f1_for_this_run = current_f1
                    logger.info(f"发现新的最佳F1分数: {best_f1_for_this_run:.4f} 在 Epoch {epoch}")
            except Exception as e:
                logger.error(f"测试模型失败: {e}")

# 训练完成，记录总结信息
logger.info("训练完成")
logger.info(f"总训练轮数: {config['epochs']}")
    # --- 单次运行结束，记录结果 ---
grid_search_results.append({
    'run_id': run_id,
    'lambda_adv': lambda_adv,
    'lambda_seg': lambda_seg,
    'lambda_ht': lambda_ht,
    'best_f1': best_f1_for_this_run
})
logger.info(f"网格搜索运行 {run_id + 1} 完成。最佳F1: {best_f1_for_this_run:.4f}")
# --- 所有网格搜索运行结束，总结并打印最佳结果 ---
logger = setup_logger('grid_search_summary') # 创建一个总结日志
logger.info("\n" + "="*50)
logger.info("网格搜索完成！总结如下：")
logger.info("="*50)
# 将结果转换为DataFrame以便查看
results_df = pd.DataFrame(grid_search_results)
logger.info("\n所有运行结果:\n" + results_df.to_string())

# 找到最佳结果
if not results_df.empty:
    best_run = results_df.loc[results_df['best_f1'].idxmax()]
    logger.info("\n" + "*"*50)
    logger.info("最佳结果:")
    logger.info(f"运行ID: {best_run['run_id']}")
    logger.info(f"最佳F1分数: {best_run['best_f1']:.4f}")
    logger.info("对应的参数:")
    logger.info(f"  lambda_adv: {best_run['lambda_adv']}")
    logger.info(f"  lambda_seg: {best_run['lambda_seg']}")
    logger.info(f"  lambda_ht: {best_run['lambda_ht']}")
    logger.info("*"*50)
else:
    logger.warning("没有有效的网格搜索结果。")
# 保存最终模型
final_model_path = "/home/my/gbznt/union_model/generator_final.pth"
final_checkpoint_path = "/home/my/gbznt/union_model/checkpoint_final.pth"
try:
    torch.save(generator.state_dict(), final_model_path)
    torch.save(discriminator.state_dict(), "/home/my/gbznt/union_model/discriminator_final.pth")
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'scheduler_G_state_dict': scheduler_G.state_dict(),
        'scheduler_D_state_dict': scheduler_D.state_dict(),
        'lr_G': optimizer_G.param_groups[0]['lr'],
        'lr_D': optimizer_D.param_groups[0]['lr'],
        'generator_union_state_dict': generator_union.state_dict(),
        'optimizer_union_G_state_dict': optimizer_G_union.state_dict(),
        #'discriminator_union_state_dict': discriminator_union.state_dict(),
        #'optimizer_union_D_state_dict': optimizer_D_union.state_dict(),
        'scheduler_union_G_state_dict': scheduler_G_union.state_dict(),
        #'scheduler_union_D_state_dict': scheduler_D_union.state_dict(),
        'lr_G_union': optimizer_G_union.param_groups[0]['lr'],
        #'lr_D_union': optimizer_D_union.param_groups[0]['lr'],
    }, final_checkpoint_path)
    logger.info(f"保存最终模型到 {final_model_path}")
    logger.info(f"保存最终检查点到 {final_checkpoint_path}")
except Exception as e:
    logger.error(f"保存最终模型失败: {e}")

# 绘制完整训练过程的损失曲线
try:
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses_fully, label='Discriminator Loss')
    plt.plot(g_losses_fully, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig(f"/home/my/gbznt/union_loss/loss_epoch_{epoch}.png")
    plt.close()
    logger.info(f"保存完整训练过程损失曲线到 /home/my/gbznt/union_loss/loss_epoch_{epoch}.png")
except Exception as e:
    logger.error(f"保存完整训练过程损失曲线失败: {e}")

# 记录硬件信息
try:
    if torch.cuda.is_available():
        logger.info(f"使用的GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        logger.info(f"GPU内存使用: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
except Exception as e:
    logger.error(f"获取硬件信息失败: {e}")