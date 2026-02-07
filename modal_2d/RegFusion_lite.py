import torch.nn as nn
from modal_2d.Restormer import Restormer
import torch.nn.functional as F
import torch
from einops import rearrange
from .classifier import VitBlock, PatchEmbedding2D
from utils_2d.warp import Warper2d, warp2D



image_warp = warp2D()
def project(x, image_size):
    """将Transformer输出投影回图像空间，功能: 将序列化的特征向量还原为2D特征图， torch.Size([1, 512, 768]) 转换成图像形状[channel, W/p, H/p]"""
    W, H = image_size[0], image_size[1]
    x = rearrange(x, 'b (w h) hidden -> b w h hidden', w=W // 16, h=H // 16)
    x = x.permute(0, 3, 1, 2)
    return x
def img_warp(flow, I):

    return Warper2d()(flow, I)
def flow_integration_ir(flow1, flow2, flow3, flow4, flow5, flow6, flow7, flow8, flow9, flow10):
    """多尺度流场融合,功能: 将不同尺度的流场上采样并加权融合"""
    # 创建上采样器
    up1 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
    up2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
    up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    # 上采样各尺度流场并加权
    flow1, flow2 = up1(flow1)*16, up1(flow2)*16
    flow3, flow4 = up2(flow3)*8, up2(flow4)*8
    flow5, flow6 = up3(flow5)*4, up3(flow6)*4
    flow7, flow8 = up4(flow7)*2, up4(flow8)*2
    flow_neg = flow1 + flow3 + flow5 + flow7 + flow9
    flow_pos = flow2 + flow4 + flow6 + flow8 + flow10
    flow = flow_pos - flow_neg  # 最终流场
    return flow, flow_neg, flow_pos

def reg(flow, feature):
    feature = Warper2d()(flow, feature)
    return feature

class model_classifer_lite(nn.Module):
    """轻量级ViT分类器 (用于特征分类)"""
    def __init__(self, in_c, num_heads, num_vit_blk, img_size, patch_size):
        super(model_classifer_lite, self).__init__()
        self.hidden_size = 256
        self.embedding = PatchEmbedding2D(in_c = in_c, embedding_dim=256, patch_size=patch_size)

        self.vit_blks = nn.Sequential()  #添加n个vit块
        for i in range(num_vit_blk):
            self.vit_blks.add_module(
                name=f'vit{i}',
                module=VitBlock(hidden_size=256,
                         num_heads=num_heads,
                         vit_drop=0.1, qkv_bias=False,
                         mlp_dim=256, mlp_drop=0.0)
            )
        self.norm = nn.LayerNorm(256)
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


class Classifier_lite(nn.Module):
    """轻量级ViT分类器 (用于图像分类)"""
    def __init__(self, in_c, num_heads, num_vit_blk, img_size, patch_size):
        super(Classifier_lite, self).__init__()
        self.hidden_size = 256
        self.embedding = PatchEmbedding2D(in_c = in_c, embedding_dim=256, patch_size=patch_size)

        self.vit_blks = nn.Sequential()  #添加n个vit块
        for i in range(num_vit_blk):
            self.vit_blks.add_module(
                name=f'vit{i}',
                module=VitBlock(hidden_size=256,
                         num_heads=num_heads,
                         vit_drop=0.1, qkv_bias=False,
                         mlp_dim=256, mlp_drop=0.0)
            )
        self.norm = nn.LayerNorm(256)
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


class Transfer(nn.Module):
    """特征转换模块 (用于跨模态特征转换)"""
    def __init__(self, num_vit, num_heads):
        super(Transfer, self).__init__()
        self.num_vit = num_vit
        self.num_heads = num_heads
        self.hidden_dim = 256
        self.cls1 = nn.Parameter(torch.zeros(1, 1, 256))
        self.cls2 = nn.Parameter(torch.zeros(1, 1, 256))
        self.VitBLK1 = nn.Sequential()
        for i in range(self.num_vit):
            self.VitBLK1.add_module(name=f'vit{i}',
                                    module=VitBlock(hidden_size=self.hidden_dim,
                                                    num_heads=self.num_heads,
                                                    vit_drop=0.0,
                                                    qkv_bias=False,
                                                    mlp_dim=256,
                                                    mlp_drop=0.0))
        self.VitBLK2 = nn.Sequential()
        for i in range(self.num_vit):
            self.VitBLK2.add_module(name=f'vit{i}',
                                    module=VitBlock(hidden_size=self.hidden_dim,
                                                    num_heads=self.num_heads,
                                                    vit_drop=0.0,
                                                    qkv_bias=False,
                                                    mlp_dim=256,
                                                    mlp_drop=0.0))
    def forward(self, x1, x2, cls1, cls2):
        """
               跨模态特征转换
               参数:
                   x1: 模态1特征序列
                   x2: 模态2特征序列
                   cls1: 模态1分类令牌
                   cls2: 模态2分类令牌
               返回:
                   x1: 转换后的模态1特征
                   x2: 转换后的模态2特征
                   new_cls1: 新的模态1分类令牌
                   new_cls2: 新的模态2分类令牌
        """
        cls1, cls2 = cls1.unsqueeze(dim=1), cls2.unsqueeze(dim=1)
        cls1 = cls1.expand(-1, x1.shape[1], -1)
        cls2 = cls2.expand(-1, x1.shape[1], -1)
        x1, x2 = x1+cls2, x2 + cls1
        class_token1 = self.cls1.expand(x1.shape[0], -1, -1)
        class_token2 = self.cls2.expand(x1.shape[0], -1, -1)
        # x1, x2 = self.MLP1(x1), self.MLP2(x2)
        x1 = torch.cat((x1, class_token1), dim=1)
        x2 = torch.cat((x2, class_token2), dim=1)
        x1 = self.VitBLK1(x1)
        x2 = self.VitBLK2(x2)
        class_token1 = x1[:, 0, :]
        class_token2 = x2[:, 0, :]
        return  x1[:, 1:, :], x2[:, 1:, :], class_token1, class_token2


class Encoder(nn.Module):
    """特征编码器 (使用Restormer块)"""
    def __init__(self):
        super(Encoder, self).__init__()
        self.rb1 = Restormer(1, 8)
        self.rb2 = Restormer(8, 3)

    def forward(self, img):
        f = self.rb1(img)
        f_ = self.rb2(f)
        return f, f_

class ModelTransfer_lite(nn.Module):
    def __init__(self, num_vit, num_heads, img_size):
        super(ModelTransfer_lite, self).__init__()
        self.img_size = img_size
        self.transfer = Transfer(num_vit=num_vit, num_heads=num_heads)
        self.classifier = Classifier_lite(in_c=3, num_heads=4, num_vit_blk=2, img_size=self.img_size, patch_size=16)
        self.modal_dis = model_classifer_lite(in_c=3, num_heads=4, num_vit_blk=2, img_size=self.img_size, patch_size=16)

    def forward(self, img1, img2):

        pre1, cls1, x1_ = self.classifier(img1)
        pre2, cls2, x2_ = self.classifier(img2)
        x1, x2, new_cls1, new_cls2 = self.transfer(x1_, x2_, cls1, cls2)
        feature_pred1, _, _ = self.modal_dis(x1)
        feature_pred2, _, _ = self.modal_dis(x2)
        return  pre1, pre2, feature_pred1, feature_pred2, x1, x2, x1_, x2_  # 分类器预测结果，特征转换器分类结果


class CrossAttention(nn.Module):
    """跨模态注意力模块"""
    def __init__(self, in_channel, out_channel):
        super(CrossAttention, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1),
                                   # nn.InstanceNorm2d(in_channel),
                                   nn.ReLU(),
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1),
                                   # nn.InstanceNorm2d(in_channel),
                                   nn.ReLU(),
                                   )
    def forward(self, f1, f2):
        f1_hat = f1# 保留原始特征
        f1 = self.conv1(f1)
        f2 = self.conv2(f2)
        att_map = f1 * f2
        att_shape = att_map.shape
        att_map = torch.reshape(att_map, [att_shape[0], att_shape[1], -1])
        att_map = F.softmax(att_map, dim=2)
        att_map = torch.reshape(att_map, att_shape)
        f1 = f1 * att_map
        f1 = f1 + f1_hat
        return f1

class ResBlk(nn.Module):
    def __init__(self, in_channel):
        super(ResBlk, self).__init__()
        self.feature_output = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )
    def forward(self ,x):
        return x + self.feature_output(x)

class FusionRegBlk_lite(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FusionRegBlk_lite, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(kernel_size=1, stride=1, padding=0, in_channels=in_channel * 2, out_channels=in_channel),
            nn.LeakyReLU())

        self.crossAtt1 = CrossAttention(in_channel, out_channel)
        # self.crossAtt2 = CrossAttention(in_channel, out_channel)
        self.feature_output = nn.Sequential(
            ResBlk(in_channel),
        )

        self.flow_output = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=2, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
            # nn.Conv2d(2, 2, 1, 1, 0),
        )
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding=0, stride=1),
                                 nn.LeakyReLU(),)

    def forward(self, f1, f2): # f2是cat后的特征

        f2 = self.conv1x1(f2)
        f1 = self.crossAtt1(f1, f2) + self.crossAtt1(f2, f1)
        f1 = self.feature_output(f1)
        f2 = self.flow_output(f1)  # 从此开始f2是flow
        f1 = self.up1(f1)
        return f1, f2


class UpBlk(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpBlk, self).__init__()
        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=1),
        )
        self.conv1 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1, stride=1)
        self.in1 = nn.InstanceNorm2d(num_features=out_c)


    def forward(self, x):
        x = self.up_sample(x)
        x = self.conv1(x)
        x = self.in1(x)
        return F.leaky_relu(x)

class Decoder(nn.Module):
    """特征解码器"""
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.channels = channels

        self.up1 = UpBlk(self.channels[0], self.channels[1])
        self.up2 = UpBlk(self.channels[1], self.channels[2])
        self.up3 = UpBlk(self.channels[2], self.channels[3])
        self.up4 = UpBlk(self.channels[3], self.channels[4])

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================
# 1) ED-style Entropy Map (soft-histogram entropy conv)
# ============================
class EntropyConv2D(nn.Module):
    """
    ED 思路：在每个像素邻域内估计灰度分布 p(x_i)，再算 Shannon entropy。
    这里用“软直方图(soft binning)+卷积聚合”实现，GPU 友好。
    - 非学习：无可训练参数（bin_centers 作为 buffer）
    """
    def __init__(self, num_bins: int = 16, win_size: int = 9, sigma: float = 0.08, eps: float = 1e-8):
        super().__init__()
        assert win_size % 2 == 1, "win_size must be odd"
        self.num_bins = num_bins
        self.win_size = win_size
        self.sigma = sigma
        self.eps = eps

        # bin centers in [0,1]
        centers = torch.linspace(0.0, 1.0, steps=num_bins).view(1, num_bins, 1, 1)
        self.register_buffer("bin_centers", centers, persistent=False)

        # group conv kernel: [num_bins, 1, ws, ws]
        k = torch.ones(num_bins, 1, win_size, win_size)
        self.register_buffer("kernel", k, persistent=False)

    @torch.no_grad()
    def _check_range(self, x):
        # 仅用于调试：输入最好已归一化到 [0,1]
        if x.min() < -1e-3 or x.max() > 1.0 + 1e-3:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,1,H,W] 期望范围 [0,1]
        return: entropy map [B,1,H,W]
        """
        # self._check_range(x)  # 可打开调试
        x = x.clamp(0.0, 1.0)

        # soft assignment to bins (Gaussian RBF)
        # w: [B, num_bins, H, W]
        diff = x - self.bin_centers  # broadcast
        w = torch.exp(-0.5 * (diff / (self.sigma + self.eps)) ** 2)

        # normalize per-pixel so sum_k w_k = 1
        w = w / (w.sum(dim=1, keepdim=True) + self.eps)

        pad = self.win_size // 2
        w_pad = F.pad(w, (pad, pad, pad, pad), mode="reflect")

        # local counts per bin via group conv
        # out: [B, num_bins, H, W]
        local_counts = F.conv2d(w_pad, self.kernel, groups=self.num_bins)

        # local prob
        area = float(self.win_size * self.win_size)
        p = local_counts / area
        p = p.clamp_min(self.eps)

        # Shannon entropy
        H = -(p * torch.log2(p)).sum(dim=1, keepdim=True)  # [B,1,H,W]
        return H


class EntropyDiffWeight(nn.Module):
    """
    改进点（解决 W 稀疏/全黑）：
    1) 输入自动 min-max 到 [0,1]（防止你数据不严格归一化导致熵异常）
    2) g 不再用 min(H1,H2)（CT脑内熵低会把全脑压死） -> 改为 mean(H1,H2)
    3) tau 默认更“宽松”，并对 w 做 pow(<1) 软化
    4) W 加 floor（地板值），防止 flow/attention 梯度被完全杀死
    5) 可选轻度平滑，避免 W 只有细线
    """
    def __init__(
        self,
        num_bins=16,
        win_size=9,
        sigma=0.08,
        tau=0.60,          # ✅ 从 0.25 放宽到 0.60（更不容易全压死）
        w_floor=0.15,      # ✅ W 地板值，保证全图最少有 0.15 的“可动性”
        w_gamma=0.5,       # ✅ w = exp(-dH/tau) ** w_gamma，gamma<1 更软
        smooth_kernel=3,   # ✅ 轻度平滑 W，减少“边缘线条图”
        eps=1e-8
    ):
        super().__init__()
        self.entropy = EntropyConv2D(num_bins=num_bins, win_size=win_size, sigma=sigma, eps=eps)
        self.tau = tau
        self.w_floor = w_floor
        self.w_gamma = w_gamma
        self.smooth_kernel = smooth_kernel
        self.eps = eps

    def _to01(self, x: torch.Tensor) -> torch.Tensor:
        # per-sample min-max to [0,1]
        mn = torch.amin(x, dim=(2, 3), keepdim=True)
        mx = torch.amax(x, dim=(2, 3), keepdim=True)
        x = (x - mn) / (mx - mn + self.eps)
        return x.clamp(0.0, 1.0)

    def _norm01(self, x: torch.Tensor) -> torch.Tensor:
        mn = torch.amin(x, dim=(2, 3), keepdim=True)
        mx = torch.amax(x, dim=(2, 3), keepdim=True)
        return (x - mn) / (mx - mn + self.eps)

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        if self.smooth_kernel is None or self.smooth_kernel <= 1:
            return x
        k = self.smooth_kernel
        pad = k // 2
        return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel_size=k, stride=1)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        """
        img1,img2: [B,1,H,W]
        return:
          W: [B,1,H,W]
          (H1,H2,dH): 便于可视化
        """
        img1 = self._to01(img1)
        img2 = self._to01(img2)

        H1 = self.entropy(img1)
        H2 = self.entropy(img2)
        dH = (H1 - H2).abs()

        # consistency gate (softened)
        w = torch.exp(-dH / (self.tau + self.eps))
        w = w.clamp(0.0, 1.0).pow(self.w_gamma)

        # information gate: ✅ mean 而不是 min（避免 CT 脑内低熵把 W 全压死）
        g_raw = 0.5 * (H1 + H2)
        g = self._norm01(g_raw)

        W = (w * g).clamp(0.0, 1.0)

        # ✅ 地板值：防止 W 大面积为 0
        if self.w_floor is not None and self.w_floor > 0:
            W = W * (1.0 - self.w_floor) + self.w_floor

        # ✅ 平滑：让 W 不只是一圈边缘线
        W = self._smooth(W).clamp(0.0, 1.0)

        return W, (H1, H2, dH)


class EntropyAwareCrossAttention(nn.Module):
    """
    改进点：
    - 仍然使用 log(W) bias，但因为 W 有 floor，不会出现大面积 -inf，把注意力彻底压死。
    - 默认把 w_beta 稍微调小更稳（你也可在 RegNet 里传参）
    """
    def __init__(self, in_channel, w_beta: float = 0.6, eps: float = 1e-8):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        self.w_beta = w_beta
        self.eps = eps

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, W: torch.Tensor):
        f1_hat = f1
        f1 = self.conv1(f1)
        f2 = self.conv2(f2)

        att_logits = f1 * f2  # [B,C,H,W]
        B, C, H, Wsp = att_logits.shape
        att_logits = att_logits.view(B, C, -1)  # [B,C,N]

        # reliability bias (W already floored)
        bias = torch.log(W.view(B, 1, -1).clamp_min(self.eps))  # [B,1,N]
        att_logits = att_logits + self.w_beta * bias

        att = F.softmax(att_logits, dim=2).view(B, C, H, Wsp)
        f1 = f1 * att + f1_hat
        return f1


class EntropyFusionRegBlk_lite(nn.Module):
    """
    改进点（解决 flow 塌缩）：
    - 不再使用 flow = flow_pred * W （硬门控）
    - 改为：flow = flow_pred * (alpha + (1-alpha)*W)
      其中 alpha>0 保证即使 W 小也能产生形变/梯度（先“动起来”）
    """
    def __init__(self, in_channel, out_channel, w_beta: float = 0.6, flow_alpha: float = 0.25):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(kernel_size=1, stride=1, padding=0, in_channels=in_channel * 2, out_channels=in_channel),
            nn.LeakyReLU()
        )

        self.crossAtt = EntropyAwareCrossAttention(in_channel, w_beta=w_beta)
        self.feature_output = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        self.flow_output = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=2, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
        )

        self.flow_alpha = flow_alpha

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(),
        )

    def forward(self, f1, f2_cat, W_s):
        f2 = self.conv1x1(f2_cat)

        f1 = self.crossAtt(f1, f2, W_s) + self.crossAtt(f2, f1, W_s)
        f1 = f1 + self.feature_output(f1)

        flow_pred = self.flow_output(f1)

        # ✅ soft gating：保证最少 alpha 的可动性
        gate = self.flow_alpha + (1.0 - self.flow_alpha) * W_s
        flow = flow_pred * gate

        f1_up = self.up1(f1)
        return f1_up, flow



# ============================
# 0) Feature-Reliability Pre-Registration Adapter
# ============================

class SobelEdge(nn.Module):
    """用 Sobel 得到边缘强度（用于抑制 CT skull 强边缘主导，可选）"""
    def __init__(self, eps=1e-6):
        super().__init__()
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("kx", kx, persistent=False)
        self.register_buffer("ky", ky, persistent=False)
        self.eps = eps

    def forward(self, x):
        # x: [B,1,H,W]
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        g = torch.sqrt(gx * gx + gy * gy + self.eps)
        # per-sample normalize to [0,1]
        gmin = torch.amin(g, dim=(2,3), keepdim=True)
        gmax = torch.amax(g, dim=(2,3), keepdim=True)
        g = (g - gmin) / (gmax - gmin + self.eps)
        return g.clamp(0, 1)


class FeatReliabilityMap(nn.Module):
    """
    从特征 f1,f2 得到 R_feat（比 W 更“厚”，覆盖内部结构，给对齐一个“活”的梯度通道）
    R_feat = 0.5 * (cos01 + exp(-dist/tau_f))
    可选：乘一个 (1-edge)^gamma，抑制 CT skull 边缘主导
    """
    def __init__(self, tau_f=0.35, smooth_ks=5, floor=0.15, edge_gamma=1.5, use_edge_suppress=True, eps=1e-6):
        super().__init__()
        self.tau_f = tau_f
        self.smooth_ks = smooth_ks
        self.floor = floor
        self.edge_gamma = edge_gamma
        self.use_edge_suppress = use_edge_suppress
        self.eps = eps
        self.sobel = SobelEdge(eps=eps)

    def forward(self, f1, f2, img2=None):
        # f1,f2: [B,C,h,w]
        f1n = F.normalize(f1, dim=1)
        f2n = F.normalize(f2, dim=1)

        cos = (f1n * f2n).sum(dim=1, keepdim=True)              # [-1,1]
        cos01 = (cos + 1.0) * 0.5                               # [0,1]

        dist = (f1n - f2n).abs().mean(dim=1, keepdim=True)      # [0,2] roughly
        expd = torch.exp(-dist / (self.tau_f + self.eps))       # (0,1]

        R = 0.5 * (cos01 + expd)                                # [0,1] dense-ish

        # 可选：抑制 skull 边缘主导（对 CT->MRI 特别有用）
        if self.use_edge_suppress and (img2 is not None):
            edge = self.sobel(img2)                             # [B,1,H,W]
            edge_s = F.interpolate(edge, size=f1.shape[2:], mode="bilinear", align_corners=False)
            suppress = (1.0 - edge_s).clamp(0, 1) ** self.edge_gamma
            R = R * suppress

        # 平滑，让 R_feat 不呈“细线”
        if self.smooth_ks >= 3:
            k = self.smooth_ks
            R = F.avg_pool2d(R, kernel_size=k, stride=1, padding=k//2)

        # 加 floor，避免门控把一切掐死
        R = self.floor + (1.0 - self.floor) * R
        return R.clamp(0, 1)


class PreRegFeatureAdapter(nn.Module):
    """
    把 “配准需要的相关性信息” 在进入 RegNet 前注入特征：
    Z = Conv1x1([f1, f2, |f1-f2|, f1*f2]) -> 256
    delta = Conv3x3(Conv3x3(Z))
    f1 <- f1 + alpha * R_feat * delta
    f2 <- f2 + alpha * R_feat * delta
    可选：在 (1-R_feat) 区域回退一点原始特征（防止过度共享）
    """
    def __init__(self, C=256, alpha=0.6, beta=0.25):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.rel = FeatReliabilityMap(
            tau_f=0.35, smooth_ks=5, floor=0.15,
            edge_gamma=1.5, use_edge_suppress=True
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(C * 4, C, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
        )
        # 原始特征回退投影（如果你传 AU_F/BU_F 进来）
        self.raw_proj = nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0)

    def forward(self, f1, f2, raw1=None, raw2=None, img2=None):
        # 1) R_feat
        R = self.rel(f1, f2, img2=img2)            # [B,1,h,w]

        # 2) 相关性上下文
        Z = torch.cat([f1, f2, (f1 - f2).abs(), f1 * f2], dim=1)
        Z = self.fuse(Z)
        delta = self.refine(Z)

        # 3) 注入（可靠区域强注入，避免内部“没梯度”）
        f1a = f1 + self.alpha * R * delta
        f2a = f2 + self.alpha * R * delta

        # 4) 不可靠区域回退一点原始特征（可选，保护差异/病灶）
        if (raw1 is not None) and (raw2 is not None):
            inv = (1.0 - R)
            f1a = f1a + self.beta * inv * self.raw_proj(raw1)
            f2a = f2a + self.beta * inv * self.raw_proj(raw2)

        return f1a, f2a, R


# ============================
# 3) EntropyDiffRegNet_lite: 在 forward 里加入 stage0 pre-adapter
# ============================

class EntropyDiffRegNet_lite(nn.Module):
    def __init__(self,
                 entropy_bins=16, entropy_win=9, entropy_sigma=0.08,
                 tau=0.25, w_beta=1.0,
                 # —— 下面是 pre-adapter 的超参（论文可写 ablation）
                 pre_alpha=0.6, pre_beta=0.25):
        super().__init__()
        self.channels = [256, 64, 32, 16, 8, 1]

        # 0) Pre-adapter（对齐模块前置：配准专用特征预处理）
        self.pre_adapter = PreRegFeatureAdapter(C=256, alpha=pre_alpha, beta=pre_beta)

        # 1) 你原来的 entropy-diff weight
        self.weight_gen = EntropyDiffWeight(
            num_bins=entropy_bins, win_size=entropy_win, sigma=entropy_sigma, tau=tau
        )

        # 2) 你原来的多尺度双分支（保持不动）
        self.f1_FR1 = EntropyFusionRegBlk_lite(in_channel=self.channels[0], out_channel=self.channels[1], w_beta=w_beta)
        self.f1_FR2 = EntropyFusionRegBlk_lite(in_channel=self.channels[1], out_channel=self.channels[2], w_beta=w_beta)
        self.f1_FR3 = EntropyFusionRegBlk_lite(in_channel=self.channels[2], out_channel=self.channels[3], w_beta=w_beta)
        self.f1_FR4 = EntropyFusionRegBlk_lite(in_channel=self.channels[3], out_channel=self.channels[4], w_beta=w_beta)
        self.f1_FR5 = EntropyFusionRegBlk_lite(in_channel=self.channels[4], out_channel=self.channels[5], w_beta=w_beta)

        self.f2_FR1 = EntropyFusionRegBlk_lite(in_channel=self.channels[0], out_channel=self.channels[1], w_beta=w_beta)
        self.f2_FR2 = EntropyFusionRegBlk_lite(in_channel=self.channels[1], out_channel=self.channels[2], w_beta=w_beta)
        self.f2_FR3 = EntropyFusionRegBlk_lite(in_channel=self.channels[2], out_channel=self.channels[3], w_beta=w_beta)
        self.f2_FR4 = EntropyFusionRegBlk_lite(in_channel=self.channels[3], out_channel=self.channels[4], w_beta=w_beta)
        self.f2_FR5 = EntropyFusionRegBlk_lite(in_channel=self.channels[4], out_channel=self.channels[5], w_beta=w_beta)

        self.D = Decoder(self.channels)

    def _resize_W(self, W, feat):
        return F.interpolate(W, size=feat.shape[2:4], mode="bilinear", align_corners=False)

    def forward(self, f1, f2, img1, img2, return_maps: bool = False,
                # 可选：把 AU_F/BU_F（project 后）传进来用于“不可靠区回退”
                raw1=None, raw2=None, return_pre: bool = False):
        """
        f1,f2: [B,256,16,16] (project 后的特征，来自 transfer 的 feature1/feature2)
        img1,img2: [B,1,256,256]
        raw1,raw2: [B,256,16,16] (project 后的 AU_F/BU_F，可选)
        """

        # ===== stage0: Pre-adapter（先把特征变得“更配准友好”）=====
        f1, f2, R_feat = self.pre_adapter(f1, f2, raw1=raw1, raw2=raw2, img2=img2)

        # ===== stage1: 你原来的 W_full（图像级熵差可靠性）=====
        W_full, (H1, H2, dH) = self.weight_gen(img1, img2)

        # ===== 原逻辑不动：多尺度 flow =====
        W16 = self._resize_W(W_full, f1)
        f_cat = torch.cat((f1, f2), dim=1)
        f1_, flow1 = self.f1_FR1(f1, f_cat, W16)
        f2_, flow2 = self.f2_FR1(f2, f_cat, W16)

        f1 = reg(flow1, f1)
        f2 = reg(flow2, f2)
        f1 = self.D.up1(f1)
        f2 = self.D.up1(f2)

        W32 = self._resize_W(W_full, f1)
        f_cat = torch.cat((f1, f2), dim=1)
        f1_, flow3 = self.f1_FR2(f1_, f_cat, W32)
        f2_, flow4 = self.f2_FR2(f2_, f_cat, W32)

        f1 = reg(flow3, f1)
        f2 = reg(flow4, f2)
        f1 = self.D.up2(f1)
        f2 = self.D.up2(f2)

        W64 = self._resize_W(W_full, f1)
        f_cat = torch.cat((f1, f2), dim=1)
        f1_, flow5 = self.f1_FR3(f1_, f_cat, W64)
        f2_, flow6 = self.f2_FR3(f2_, f_cat, W64)

        f1 = reg(flow5, f1)
        f2 = reg(flow6, f2)
        f1 = self.D.up3(f1)
        f2 = self.D.up3(f2)

        W128 = self._resize_W(W_full, f1)
        f_cat = torch.cat((f1, f2), dim=1)
        f1_, flow7 = self.f1_FR4(f1_, f_cat, W128)
        f2_, flow8 = self.f2_FR4(f2_, f_cat, W128)

        f1 = reg(flow7, f1)
        f2 = reg(flow8, f2)
        f1 = self.D.up4(f1)
        f2 = self.D.up4(f2)

        W256 = self._resize_W(W_full, f1)
        f_cat = torch.cat((f1, f2), dim=1)
        f1_, flow9 = self.f1_FR5(f1_, f_cat, W256)
        f2_, flow10 = self.f2_FR5(f2_, f_cat, W256)

        f1 = reg(flow9, f1)
        f2 = reg(flow10, f2)

        flow, flow_neg, flow_pos = flow_integration_ir(
            flow1, flow2, flow3, flow4, flow5, flow6, flow7, flow8, flow9, flow10
        )
        flows = [flow1, flow2, flow3, flow4, flow5, flow6, flow7, flow8, flow9, flow10]

        if return_maps:
            if return_pre:
                return f1, f2, flows, flow, flow_neg, flow_pos, W_full, (H1, H2, dH), R_feat
            return f1, f2, flows, flow, flow_neg, flow_pos, W_full, (H1, H2, dH)

        if return_pre:
            return f1, f2, flows, flow, flow_neg, flow_pos, R_feat

        return f1, f2, flows, flow, flow_neg, flow_pos

# class EntropyDiffRegNet_lite(nn.Module):
#     """
#     兼容你现有训练/测试调用：
#       reg_net(feature1, feature2, img1, img2, return_maps=True)
#     返回值结构不变：仍可保存 W/H1/H2/dH
#     """
#     def __init__(
#         self,
#         entropy_bins=16,
#         entropy_win=9,
#         entropy_sigma=0.08,
#         tau=0.60,          # ✅ 同步放宽
#         w_beta=0.6,
#         w_floor=0.15,
#         w_gamma=0.5,
#         flow_alpha=0.25,
#         smooth_kernel=3
#     ):
#         super().__init__()
#         self.channels = [256, 64, 32, 16, 8, 1]
#
#         self.weight_gen = EntropyDiffWeight(
#             num_bins=entropy_bins,
#             win_size=entropy_win,
#             sigma=entropy_sigma,
#             tau=tau,
#             w_floor=w_floor,
#             w_gamma=w_gamma,
#             smooth_kernel=smooth_kernel
#         )
#
#         self.f1_FR1 = EntropyFusionRegBlk_lite(self.channels[0], self.channels[1], w_beta=w_beta, flow_alpha=flow_alpha)
#         self.f1_FR2 = EntropyFusionRegBlk_lite(self.channels[1], self.channels[2], w_beta=w_beta, flow_alpha=flow_alpha)
#         self.f1_FR3 = EntropyFusionRegBlk_lite(self.channels[2], self.channels[3], w_beta=w_beta, flow_alpha=flow_alpha)
#         self.f1_FR4 = EntropyFusionRegBlk_lite(self.channels[3], self.channels[4], w_beta=w_beta, flow_alpha=flow_alpha)
#         self.f1_FR5 = EntropyFusionRegBlk_lite(self.channels[4], self.channels[5], w_beta=w_beta, flow_alpha=flow_alpha)
#
#         self.f2_FR1 = EntropyFusionRegBlk_lite(self.channels[0], self.channels[1], w_beta=w_beta, flow_alpha=flow_alpha)
#         self.f2_FR2 = EntropyFusionRegBlk_lite(self.channels[1], self.channels[2], w_beta=w_beta, flow_alpha=flow_alpha)
#         self.f2_FR3 = EntropyFusionRegBlk_lite(self.channels[2], self.channels[3], w_beta=w_beta, flow_alpha=flow_alpha)
#         self.f2_FR4 = EntropyFusionRegBlk_lite(self.channels[3], self.channels[4], w_beta=w_beta, flow_alpha=flow_alpha)
#         self.f2_FR5 = EntropyFusionRegBlk_lite(self.channels[4], self.channels[5], w_beta=w_beta, flow_alpha=flow_alpha)
#
#         self.D = Decoder(self.channels)
#
#     def _resize_W(self, W, feat):
#         return F.interpolate(W, size=feat.shape[2:4], mode="bilinear", align_corners=False)
#
#     def forward(self, f1, f2, img1, img2, return_maps: bool = False):
#         # 1) compute W once (but now W 不会稀疏到全黑)
#         W_full, (H1, H2, dH) = self.weight_gen(img1, img2)
#
#         # stage 1
#         W16 = self._resize_W(W_full, f1)
#         f_cat = torch.cat((f1, f2), dim=1)
#         f1_, flow1 = self.f1_FR1(f1, f_cat, W16)
#         f2_, flow2 = self.f2_FR1(f2, f_cat, W16)
#
#         f1 = reg(flow1, f1)
#         f2 = reg(flow2, f2)
#         f1 = self.D.up1(f1)
#         f2 = self.D.up1(f2)
#
#         # stage 2
#         W32 = self._resize_W(W_full, f1)
#         f_cat = torch.cat((f1, f2), dim=1)
#         f1_, flow3 = self.f1_FR2(f1_, f_cat, W32)
#         f2_, flow4 = self.f2_FR2(f2_, f_cat, W32)
#
#         f1 = reg(flow3, f1)
#         f2 = reg(flow4, f2)
#         f1 = self.D.up2(f1)
#         f2 = self.D.up2(f2)
#
#         # stage 3
#         W64 = self._resize_W(W_full, f1)
#         f_cat = torch.cat((f1, f2), dim=1)
#         f1_, flow5 = self.f1_FR3(f1_, f_cat, W64)
#         f2_, flow6 = self.f2_FR3(f2_, f_cat, W64)
#
#         f1 = reg(flow5, f1)
#         f2 = reg(flow6, f2)
#         f1 = self.D.up3(f1)
#         f2 = self.D.up3(f2)
#
#         # stage 4
#         W128 = self._resize_W(W_full, f1)
#         f_cat = torch.cat((f1, f2), dim=1)
#         f1_, flow7 = self.f1_FR4(f1_, f_cat, W128)
#         f2_, flow8 = self.f2_FR4(f2_, f_cat, W128)
#
#         f1 = reg(flow7, f1)
#         f2 = reg(flow8, f2)
#         f1 = self.D.up4(f1)
#         f2 = self.D.up4(f2)
#
#         # stage 5
#         W256 = self._resize_W(W_full, f1)
#         f_cat = torch.cat((f1, f2), dim=1)
#         f1_, flow9 = self.f1_FR5(f1_, f_cat, W256)
#         f2_, flow10 = self.f2_FR5(f2_, f_cat, W256)
#
#         f1 = reg(flow9, f1)
#         f2 = reg(flow10, f2)
#
#         flow, flow_neg, flow_pos = flow_integration_ir(
#             flow1, flow2, flow3, flow4, flow5, flow6, flow7, flow8, flow9, flow10
#         )
#         flows = [flow1, flow2, flow3, flow4, flow5, flow6, flow7, flow8, flow9, flow10]
#
#         if return_maps:
#             return f1, f2, flows, flow, flow_neg, flow_pos, W_full, (H1, H2, dH)
#
#         return f1, f2, flows, flow, flow_neg, flow_pos

class UpSampler_V2(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpSampler_V2, self).__init__()
        # 特征上采样分支
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        # 融合特征上采样分支
        self.up3 =  nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, AU_F, BU_F, feature):
        """
               多分支上采样
               参数:
                   AU_F: 模态A上采样特征
                   BU_F: 模态B上采样特征
                   feature: 融合特征
               返回:
                   AU_F: 上采样后的模态A特征
                   BU_F: 上采样后的模态B特征
                   feature: 上采样后的融合特征
               """
        AU_F = self.up1(AU_F)
        BU_F = self.up1(BU_F)
        feature = self.up3(feature)
        return AU_F, BU_F, feature


class FusionNet_lite(nn.Module):
    """轻量级图像融合网络"""
    def __init__(self):
        super(FusionNet_lite, self).__init__()
        # 通道配置
        self.cn = [256, 64, 32, 16, 12, 8]
        # 特征融合块序列
        self.F1 = Restormer(in_c=self.cn[0]*2, out_c=self.cn[1])
        self.up_sample1 = UpSampler_V2(in_c=self.cn[0], out_c=self.cn[1])

        self.F2 = Restormer(in_c=self.cn[1]*3, out_c=self.cn[2])
        self.up_sample2 = UpSampler_V2(in_c=self.cn[1], out_c=self.cn[2])

        self.F3 = Restormer(in_c=self.cn[2]*3, out_c=self.cn[3])
        self.up_sample3 = UpSampler_V2(in_c=self.cn[2], out_c=self.cn[3])

        self.F4 = Restormer(in_c=self.cn[3]*3, out_c=self.cn[4])
        self.up_sample4 =nn.Upsample(scale_factor=2, mode='bilinear')
        # 输出层
        self.outLayer = nn.Sequential(Restormer(in_c=self.cn[4] + 16, out_c=self.cn[4]),
                                      Restormer(in_c=self.cn[4], out_c=1),
                                      nn.Sigmoid())

    def forward(self, AS_F, BS_F, AU_F, BU_F, flow):
        """
              多模态图像融合流程
              参数:
                  AS_F: 模态A浅层特征
                  BS_F: 模态B浅层特征
                  AU_F: 模态A上采样特征
                  BU_F: 模态B上采样特征
                  flow: 配准流场
              返回:
                  feature: 融合后的图像
         """
        # 第一级融合
        flow_d = F.upsample(flow, BU_F.size()[2:4], mode='bilinear') / 16
        BU_F_w = img_warp(flow_d, BU_F)
        feature = self.F1(torch.cat((AU_F, BU_F_w), dim=1))  # 通道数降低了,特征融合
        # 上采样并准备下一级
        AU_F, BU_F, feature = self.up_sample1(AU_F, BU_F, feature)
        # 第二级融合
        flow_d = F.upsample(flow, BU_F.size()[2:4], mode='bilinear') / 8
        BU_F_w = img_warp(flow_d, BU_F)
        feature = self.F2(torch.cat([torch.cat([AU_F, BU_F_w], dim=1), feature], dim=1))

        AU_F, BU_F, feature = self.up_sample2(AU_F, BU_F, feature)
        flow_d = F.upsample(flow, BU_F.size()[2:4], mode='bilinear') / 4
        BU_F_w = img_warp(flow_d, BU_F)
        feature = self.F3(torch.cat([torch.cat([AU_F, BU_F_w], dim=1), feature], dim=1))

        AU_F, BU_F, feature = self.up_sample3(AU_F, BU_F, feature)
        flow_d = F.upsample(flow, BU_F.size()[2:4], mode='bilinear') / 2
        BU_F_w = img_warp(flow_d, BU_F)
        feature = self.F4(torch.cat([torch.cat([AU_F, BU_F_w], dim=1), feature], dim=1))
        # 最终上采样
        feature = self.up_sample4(feature)
        # 融合浅层特征
        BS_F_w = img_warp(flow, BS_F)# 变形模态B浅层特征
        S_F = torch.cat([AS_F, BS_F_w], dim=1)# 拼接浅层特征
        # 输出融合图像
        feature = self.outLayer(torch.cat([feature, S_F], dim=1))
        return feature

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    print(param_count)






