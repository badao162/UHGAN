import os
import glob
import cv2      
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image   
from model import UNetGenerator, Discriminator, UNetGenerator_union  #, Discriminator_union
from utils import calculate_metrics, plot_comparison
import logging  
class TestDataset(Dataset):
    def __init__(self, root_dir):
        self.sat_images = sorted(glob.glob(os.path.join(root_dir, "*_sat.jpg")))
        self.mask_images = sorted(glob.glob(os.path.join(root_dir, "*_mask.png")))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.sat_images)

    def __getitem__(self, idx):
        # 读取图像
        sat_img = cv2.imread(self.sat_images[idx])
        sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)
        mask_img = cv2.imread(self.mask_images[idx], cv2.IMREAD_GRAYSCALE)

        # 转换为Tensor
        sat_tensor = self.transform(Image.fromarray(sat_img))
        mask_tensor = transforms.ToTensor()(Image.fromarray(mask_img))

        return sat_tensor, mask_tensor, self.sat_images[idx]
# 修改test_model函数，添加日志记录
def test_model(epoch, path):
    logger.info(f"开始测试模型 - Epoch {epoch}")
    os.makedirs(config["output_dir"], exist_ok=True)
    generator = UNetGenerator().to(config["device"])
    discriminator = Discriminator().to(config["device"])
    generator_union = UNetGenerator_union().to(config["device"])
    #discriminator_union = Discriminator_union().to(config["device"])
    
    # 初始化
    logger.info(f"从路径加载检查点: {path}")
    checkpoint = torch.load(path)

    # 恢复模型的状态字典
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    generator_union.load_state_dict(checkpoint['generator_union_state_dict'])
    #discriminator_union.load_state_dict(checkpoint['discriminator_union_state_dict'])
    
    dataset = TestDataset(config["test_data_path"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    total_precision = 0
    total_recall = 0
    total_f1 = 0
    count = 0
    union_total_precision = 0
    union_total_recall = 0
    union_total_f1 = 0
    union_count = 0
    
    logger.info(f"测试数据集大小: {len(dataloader)}")
    
    with torch.no_grad():
        for i, (sat_imgs, real_masks, img_paths) in enumerate(dataloader):
            sat_imgs = sat_imgs.to(config["device"])

            # 生成预测
            pred_masks = generator(sat_imgs)
            union_pred_masks = generator_union(pred_masks)

            # 转换为numpy
            pred_np = pred_masks.squeeze().cpu().numpy()
            real_np = real_masks.squeeze().cpu().numpy()
            union_pred_np = union_pred_masks.squeeze().cpu().numpy()

            # 计算指标
            precision, recall, f1 = calculate_metrics(pred_np, real_np)
            union_precision, union_recall, union_f1 = calculate_metrics(union_pred_np, real_np)
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            count += 1
            union_total_precision += union_precision
            union_total_recall += union_recall
            union_total_f1 += union_f1
            union_count += 1

            # 保存可视化结果
            img_name = os.path.basename(img_paths[0]).replace("_sat.jpg", "")
            plot_comparison(sat_imgs[0].cpu(),
                            real_masks[0].cpu(),
                            pred_masks[0].cpu(),
                            union_pred_masks[0].cpu(),
                            f"{config['test_output']}/epoch{epoch}_{img_name}_result.png")

            logger.info(f"已处理 {i + 1}/{len(dataloader)} | "
                  f"精确率: {precision:.4f} | "
                  f"召回率: {recall:.4f} | "
                  f"F1: {f1:.4f} | "
                  f"联合精确率: {union_precision:.4f} | "
                  f"联合召回率: {union_recall:.4f} | "
                  f"联合F1: {union_f1:.4f}")

    # 计算平均指标
    avg_precision = total_precision / count
    avg_recall = total_recall / count
    avg_f1 = total_f1 / count

    union_avg_precision = union_total_precision / union_count
    union_avg_recall = union_total_recall / union_count
    union_avg_f1 = union_total_f1 / union_count

    logger.info("\n最终评估指标:")
    logger.info(f"平均精确率: {avg_precision:.4f}")
    logger.info(f"平均召回率: {avg_recall:.4f}")
    logger.info(f"平均F1分数: {avg_f1:.4f}")

    logger.info(f"\n联合模型最终评估指标:")
    logger.info(f"联合平均精确率: {union_avg_precision:.4f}")
    logger.info(f"联合平均召回率: {union_avg_recall:.4f}")
    logger.info(f"联合平均F1分数: {union_avg_f1:.4f}")
    
    return {"precision": avg_precision, "recall": avg_recall, "f1": avg_f1, 
            "union_precision": union_avg_precision, "union_recall": union_avg_recall, "union_f1": union_avg_f1}
