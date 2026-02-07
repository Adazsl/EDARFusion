import torch
import torch.nn as nn
import torch.nn.functional as F

#from functorch.einops import rearrange
from einops import rearrange


class MDTA(nn.Module):
    """多深度卷积转置注意力模块 (Multi-Dconv Head Transposed Attention)
       功能：在通道维度上应用自注意力机制，捕获全局通道依赖关系
       图像恢复应用：增强重要特征通道，抑制噪声通道
       """
    def __init__(self, out_c):#out_c (int): 输出通道数
        super(MDTA, self).__init__()
        # 查询(Q)分支: 1x1卷积 + 3x3深度卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, 1, 0),# 1x1卷积融合通道信息
            nn.Conv2d(out_c, out_c, 3, 1, 1)# 3x3卷积捕获空间上下文
        )
        # 键(K)分支: 1x1卷积 + 3x3深度卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, 1, 0),
            nn.Conv2d(out_c, out_c, 3, 1, 1)
        )
        # 值(V)分支: 1x1卷积 + 3x3深度卷积
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, 1, 0),
            nn.Conv2d(out_c, out_c, 3, 1, 1)
        )
        # 输出投影层
        self.conv4 = nn.Conv2d(out_c, out_c, 1, 1, 0)
    def forward(self, x):
        """前向传播,参数: x (torch.Tensor): 输入特征图 [B, C, H, W],返回:torch.Tensor: 注意力增强的特征图"""
        x_o = x  # 保存残差连接
        x = F.layer_norm(x, x.shape[-2:])   # 层归一化 (在空间维度上),输入: [B, C, H, W], 归一化维度: [H, W]
        C , W, H = x.size()[1], x.size()[2], x.size()[3]
        # 计算查询(Q)
        q = self.conv1(x)
        q = rearrange(q, 'b c w h -> b (w h) c')
        # 计算键(K)
        k = self.conv2(x)
        k = rearrange(k, 'b c w h -> b c (w h)')
        v = self.conv3(x)
        v = rearrange(v, 'b c w h -> b (w h) c')
        # 计算注意力图: A = softmax(K^T * Q)
        A = torch.matmul(k, q)   # [B, C, C] (通道注意力图)
        A = rearrange(A, 'b c1 c2 -> b (c1 c2)', c1=C, c2=C)  # 展平为向量 [B, C*C]
        A = torch.softmax(A, dim=1)  # 通道维度归一化
        A = rearrange(A, 'b (c1 c2) -> b c1 c2', c1=C, c2=C)  # 恢复为矩阵 [B, C, C]
        # 应用注意力: V' = V * A
        v = torch.matmul(v, A)   # [B, N, C] * [B, C, C] -> [B, N, C]
        v = rearrange(v, 'b (h w) c -> b c h w', c = C, h=H, w=W) # 恢复空间维度,[B, C, H, W]
        return self.conv4(v) + x_o # 输出投影+残差连接

class GDFN(nn.Module):
    """门控深度前馈网络 (Gated-Dconv Feed-Forward Network)
        功能：提供非线性特征变换能力
        图像恢复应用：增强特征表达能力，抑制噪声
    """
    def __init__(self, out_c):
        super(GDFN, self).__init__()
        # 门控分支1: 1x1卷积扩展通道 + 3x3深度卷积
        self.Dconv1 = nn.Sequential(
            nn.Conv2d(out_c, out_c*4, 1, 1, 0),# 通道扩展4倍
            nn.Conv2d(out_c*4, out_c*4, 3, 1, 1) #空间特征提取
        )
        # 门控分支2: 1x1卷积扩展通道 + 3x3深度卷积
        self.Dconv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c * 4, 1, 1, 0),
            nn.Conv2d(out_c * 4, out_c * 4, 3, 1, 1)
        )
        # 输出投影层 (压缩通道)
        self.conv = nn.Conv2d(out_c * 4, out_c, 1, 1, 0)
    def forward(self, x):
        """前向传播,参数:x (torch.Tensor): 输入特征图 [B, C, H, W],返回: torch.Tensor: 非线性变换后的特征图"""
        x_o = x
        x = F.layer_norm(x, x.shape[-2:])
        # 门控机制: G = Gelu(Dconv1(x)) ⊙ Dconv2(x)
        x = F.gelu(self.Dconv1(x)) * self.Dconv2(x)
        x = x_o + self.conv(x)
        return x


class Restormer(nn.Module):
    """Restormer 基础块,功能：结合MDTA和GDFN的图像恢复基础模块"""
    def __init__(self, in_c, out_c):
        super(Restormer, self).__init__()
        self.mlp = nn.Conv2d(in_c, out_c, 1, 1, 0) # 通道调整层 (输入通道 -> 输出通道)
        self.mdta = MDTA(out_c)# 多深度卷积转置注意力模块
        self.gdfn = GDFN(out_c) # 门控深度前馈网络
    def forward(self, feature):
        feature = self.mlp(feature)# 通道调整
        feature = self.mdta(feature) # 通道注意力增强
        return self.gdfn(feature) # 非线性变换