import monai  # 医学图像分析库
import torch
from einops import rearrange
from monai.networks.blocks import SABlock, MLPBlock # MONAI自注意力块和MLP块
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock # 图像块嵌入模块
import torch.nn as nn
import torch.nn.functional as F



def project(x, image_size):
    """将序列化的图像块特征还原为2D特征图,torch.Size([1, 512, 768]) 转换成图像形状[channel, W/p, H/p]"""
    W, H = image_size[0], image_size[1]
    x = rearrange(x, 'b (w h) hidden -> b w h hidden', w=W // 16, h=H // 16)# 重排维度: [B, (w h), hidden] -> [B, w, h, hidden]
    x = x.permute(0, 3, 1, 2) # 调整维度顺序: [B, w, h, hidden] -> [B, hidden, w, h]
    return x


class PatchEmbedding2D(nn.Module):
    """自定义2D图像块嵌入层（带位置编码），patch"""
    def __init__(self, in_c, embedding_dim, patch_size):#in_c (int): 输入通道数,embedding_dim (int): 嵌入维度, patch_size (int): patch图像块大小
        super(PatchEmbedding2D, self).__init__()
        self.patch_embedding = nn.Conv2d(in_c, embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0)    # 使用卷积实现图像块嵌入,如由（3，224，224）->（768，14，14）
        self.position_embedding = nn.Parameter(torch.randn([1, 1, 256, 256]))# 可学习的位置编码 (初始化为随机值)
    def forward(self, x):
        x_H, x_W = x.shape[2], x.shape[3]
        # 插值调整位置编码尺寸以匹配输入图像
        position_embedding = F.interpolate(self.position_embedding, size=[x_H, x_W], mode='bilinear', align_corners=True)
        x = x + position_embedding[:, :]# 添加位置编码
        x = self.patch_embedding(x)  # 应用图像块嵌入卷积,输出b, embedding_dim, H/16, W/16
        x = rearrange(x, 'b embedding_dim h w -> b (h w) embedding_dim')
        return x #转换成了token


class VitBlock(nn.Module):
    """ViT基础块，Attention"""
    def __init__(self, hidden_size, num_heads, vit_drop, qkv_bias, mlp_dim, mlp_drop):
        """
               参数:
                   hidden_size (int): 隐藏层维度
                   num_heads (int): 注意力头数
                   vit_drop (float): 注意力层dropout率，防止过拟合
                   qkv_bias (bool): 是否在QKV投影中使用偏置
                   mlp_dim (int): MLP隐藏层维度
                   mlp_drop (float): MLP层dropout率
               """
        super(VitBlock, self).__init__()
        self.attention = SABlock(hidden_size=hidden_size, num_heads=num_heads, dropout_rate=vit_drop, qkv_bias=qkv_bias)# 自注意力块
        self.mlp = MLPBlock(hidden_size=hidden_size, mlp_dim=mlp_dim, dropout_rate=mlp_drop) # 多层感知机块
        self.norm_layer1 = nn.LayerNorm(hidden_size)  # 层归一化
        self.norm_layer2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x + self.attention(self.norm_layer2(x))  # 自注意力残差连接 batch patch emb_dim
        x = x + self.mlp(self.norm_layer2(x)) # MLP残差连接
        return x


class VIT(nn.Module):
    """完整Vision Transformer模型 (使用MONAI的嵌入块)"""
    def __init__(self, in_c, num_heads, num_vit_blk, img_size, patch_size):
        """
               参数:
                   in_c (int): 输入通道数
                   num_heads (int): 注意力头数
                   num_vit_blk (int): ViT块数量
                   img_size (tuple): 图像尺寸 (H, W)
                   patch_size (int): 图像块大小
               """
        super(VIT, self).__init__()
        self.hidden_size = 768
        self.embedding = PatchEmbeddingBlock(in_channels=in_c,
                                             img_size=img_size,
                                             patch_size=patch_size,
                                             hidden_size=768,
                                             num_heads=num_heads,
                                             pos_embed='perceptron', # 感知机位置编码
                                             dropout_rate=0.0,
                                             spatial_dims=2) # 2D图像


        self.vit_blks = nn.Sequential()  #添加n个vit块
        for i in range(num_vit_blk):
            self.vit_blks.add_module(
                name=f'vit{i}',
                module=VitBlock(hidden_size=768,
                         num_heads=num_heads,
                         vit_drop=0.1, qkv_bias=False,
                         mlp_dim=3072, mlp_drop=0.0)# MLP扩展因子4 (768*4=3072)
            )
            # 分类头
        self.norm = nn.LayerNorm(768)
        self.head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                  nn.GELU(),# 高斯误差线性单元激活
                                  nn.Linear(self.hidden_size, 2))# 二分类
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size)) # 分类令牌 (CLS token)

    def forward(self, x):
        """前向传播，返回:predict: 分类预测 [B, 2]，class_token: CLS令牌特征 [B, hidden_size]，patch_features: 图像块特征 [B, num_patches-1, hidden_size] """
        x = self.embedding(x)  # image_embedding，图像块嵌入
        class_token = self.cls_token.expand(x.shape[0], -1, -1)# 添加CLS令牌
        x = torch.cat((class_token, x), dim=1)
        x = self.vit_blks(x)  # 通过ViT块

        class_token = x[:, 0]
        predict = self.head(class_token)# 分类预测
        return predict, class_token, x[:, 1:]

class model_classifer(nn.Module):
    """分类模型 (基于ViT特征)"""
    def __init__(self, in_c, num_heads, num_vit_blk, img_size, patch_size): #注意：此模块接受预提取的图像块特征作为输入
        super(model_classifer, self).__init__()
        self.hidden_size = 768
        self.embedding = PatchEmbeddingBlock(in_channels=in_c,
                                             img_size=img_size,
                                             patch_size=patch_size,
                                             hidden_size=768,
                                             num_heads=num_heads,
                                             pos_embed='perceptron',
                                             dropout_rate=0.0,
                                             spatial_dims=2)

        self.vit_blks = nn.Sequential()  #添加n个vit块
        for i in range(num_vit_blk):
            self.vit_blks.add_module(
                name=f'vit{i}',
                module=VitBlock(hidden_size=768,
                         num_heads=num_heads,
                         vit_drop=0.1, qkv_bias=False,
                         mlp_dim=3072, mlp_drop=0.0)
            )
            # 分类头
        self.norm = nn.LayerNorm(768)
        self.head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                  nn.GELU(),
                                  nn.Linear(self.hidden_size, 2))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

    def forward(self, x):#前向传播 (输入为图像块特征序列)
        class_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.vit_blks(x)

        class_token = x[:, 0]
        predict = self.head(class_token)
        return predict, class_token, x[:, 1:]


class VIT_V2(nn.Module):
    """ViT模型版本2 (使用自定义嵌入块)"""
    def __init__(self, in_c, num_heads, num_vit_blk, img_size, patch_size):
        super(VIT_V2, self).__init__()
        self.hidden_size = 768
        self.embedding = PatchEmbedding2D(in_c = in_c, embedding_dim=768, patch_size=patch_size)

        self.vit_blks = nn.Sequential()  #添加n个vit块
        for i in range(num_vit_blk):
            self.vit_blks.add_module(
                name=f'vit{i}',
                module=VitBlock(hidden_size=768,
                         num_heads=num_heads,
                         vit_drop=0.1, qkv_bias=False,
                         mlp_dim=3072, mlp_drop=0.0)
            )
        self.norm = nn.LayerNorm(768)
        self.head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                  nn.GELU(),
                                  nn.Linear(self.hidden_size, 2))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

    def forward(self, x):
        x = self.embedding(x)  # image_embedding
        class_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.vit_blks(x)
        class_token = x[:, 0]
        predict = self.head(class_token)
        return predict, class_token, x[:, 1:]


class model_classifer_V2(nn.Module):
    """分类模型版本2 (使用自定义嵌入块特征)"""
    def __init__(self, in_c, num_heads, num_vit_blk, img_size, patch_size):
        super(model_classifer_V2, self).__init__()
        self.hidden_size = 768
        self.embedding = PatchEmbedding2D(in_c = in_c, embedding_dim=768, patch_size=patch_size)

        self.vit_blks = nn.Sequential()  #添加n个vit块
        for i in range(num_vit_blk):
            self.vit_blks.add_module(
                name=f'vit{i}',
                module=VitBlock(hidden_size=768,
                         num_heads=num_heads,
                         vit_drop=0.1, qkv_bias=False,
                         mlp_dim=3072, mlp_drop=0.0)
            )
        self.norm = nn.LayerNorm(768)
        self.head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                  nn.GELU(),
                                  nn.Linear(self.hidden_size, 2))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

    def forward(self, x):
        class_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.vit_blks(x)
        class_token = x[:, 0]
        predict = self.head(class_token)
        return predict, class_token, x[:, 1:]
