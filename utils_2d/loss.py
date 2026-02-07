import torch
from torch import nn
from monai.losses.ssim_loss import SSIMLoss
import torch.nn.functional as F


class Sobelxy(nn.Module):
    """Sobel边缘检测算子模块

     功能：计算图像的梯度强度图
     医学应用：突出图像中的解剖结构边缘
     """
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],  # 定义Sobel X方向卷积核
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        # 注册为不可训练参数
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1) # X方向梯度
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)# 梯度幅值

def L1_loss(tensor1, tensor2):
    """L1损失函数（平均绝对误差）
       参数:
           tensor1 (torch.Tensor): 预测张量
           tensor2 (torch.Tensor): 目标张量"""
    loss = nn.L1Loss()
    l = loss(tensor1, tensor2)
    return l

def r_loss(flow):
    """流场平滑正则化损失

    功能：惩罚流场的剧烈变化，确保变形平滑连续
    医学意义：避免不现实的器官形变
    参数:
        flow (torch.Tensor): 流场张量 [B, 2, H, W]
    返回:
        torch.Tensor: 平滑损失值
    """
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
    dx = dx * dx
    dy = dy * dy
    d = torch.mean(dx) + torch.mean(dy)
    grad = d / 3.0
    return grad

def ssim_loss(img1, img2):
    """结构相似性损失 (SSIM)
      功能：衡量两幅图像在结构信息上的相似度
      医学意义：保持解剖结构一致性
      参数:
          img1 (torch.Tensor): 图像1 [B, C, H, W]
          img2 (torch.Tensor): 图像2 [B, C, H, W]
      返回:
          torch.Tensor: SSIM损失值
      """
    # img1 = normalize(img1)
    # img2 = normalize(img2)
    # print(img1.shape)
    # print(img2.shape)
    device = img1.device
    data_range = torch.tensor(1.0).to(device)
    return SSIMLoss(spatial_dims=2)(img1, img2)

def gradient_loss(fusion_img, img1, img2):
    """融合图像梯度一致性损失
       功能：确保融合图像保留输入图像的边缘特征
       医学意义：保持关键解剖结构的清晰度"""
    grad_filter = Sobelxy().requires_grad_(False)
    grad_filter.to(fusion_img.device)
    fusion_img_g = grad_filter(fusion_img)
    max_g_img1_2 = torch.maximum(grad_filter(img1), grad_filter(img2))
    return L1_loss(fusion_img_g, max_g_img1_2)

def regFusion_loss(label1, label2,
                   pre1, pre2,
                   feature_pred1, feature_pred2,
                   flow, flows, warped_img2, flow_GT,
                   img1, img1_2, fusion_img,
                   parameter):
    # 1. 模态分类损失
    cls_loss = nn.CrossEntropyLoss()(pre1, label1) +  nn.CrossEntropyLoss()(pre2, label2)
    # 2. 特征转换损失
    grad_filter = Sobelxy().requires_grad_(False)
    grad_filter.to(fusion_img.device)
    trans_label = torch.tensor([0.5, 0.5]).expand(feature_pred1.shape[0], -1).to(feature_pred1.device)
    transfer_loss = nn.CrossEntropyLoss()(feature_pred1, trans_label)  + nn.CrossEntropyLoss()(feature_pred2, trans_label)
    # 3. 流场平滑正则化损失 (多尺度)
    flow_loss = torch.tensor(0.0).to(feature_pred1.device)
    alpha = 0.0001# 尺度权重
    for i in range(len(flows) // 2):
        flow_loss += (r_loss(flows[i]) + r_loss(flows[i + 1])) * alpha
        alpha *= 10# 增大深层权重
    flow_loss += r_loss(flow)# 最终流场
    # 4. 图像融合质量损失
    ssim1 = ssim_loss(fusion_img, img1)# 融合图与参考图相似度
    ssim2 = ssim_loss(fusion_img, warped_img2)# 融合图与变形图相似度
    fu_loss = ssim1 + parameter*ssim2 + 0.5*L1_loss(fusion_img, torch.maximum(img1, warped_img2)) + gradient_loss(fusion_img, img1, warped_img2)
    # 5. 配准精度损失
    reg_loss = ssim_loss(img1_2, warped_img2) + L1_loss(img1_2, warped_img2)

    return cls_loss, transfer_loss, flow_loss, fu_loss, reg_loss, ssim1, ssim2
