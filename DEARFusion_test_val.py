import os
import csv
import numpy as np
import torch
import warnings
import torch.nn.functional as F

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.BrainDataset_2D import TestData
from utils_2d.warp import Warper2d, warp2D
from modal_2d.RegFusion_lite import Encoder, ModelTransfer_lite, EntropyDiffRegNet_lite, FusionNet_lite
from utils_2d.utils import project, rgb2ycbcr, ycbcr2rgb
from utils_2d.metric1 import compute_all_metrics


def validate_mask(encoder, transfer, reg_net, fusion_net, dataloader, modal):
    def norm01(x, eps=1e-8):
        x_min = torch.amin(x, dim=(2, 3), keepdim=True)
        x_max = torch.amax(x, dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)

    epoch_iterator = tqdm(dataloader, desc='Val (X / X Steps)', ncols=150, leave=True, position=0)
    encoder.eval()
    transfer.eval()
    reg_net.eval()
    fusion_net.eval()

    figure_save_path = f"./test_fusionResult/{modal}_result_MRI"
    if not os.path.exists(figure_save_path):
        os.makedirs(os.path.join(figure_save_path, "MRI"))
        os.makedirs(os.path.join(figure_save_path, f"{modal}"))
        os.makedirs(os.path.join(figure_save_path, "Fusion"))
        os.makedirs(os.path.join(figure_save_path, f"{modal}_align"))
        os.makedirs(os.path.join(figure_save_path, f"{modal}_label"))

        os.makedirs(os.path.join(figure_save_path, f"{modal}_W"), exist_ok=True)
        os.makedirs(os.path.join(figure_save_path, f"{modal}_H1"), exist_ok=True)
        os.makedirs(os.path.join(figure_save_path, f"{modal}_H2"), exist_ok=True)
        os.makedirs(os.path.join(figure_save_path, f"{modal}_dH"), exist_ok=True)
        os.makedirs(os.path.join(figure_save_path, f"{modal}_R_feat"), exist_ok=True)

    image_warp = Warper2d()
    device = torch.device('cpu')

    metric_records = []
    with torch.no_grad():
        for i, batch in enumerate(epoch_iterator):
            img1, img2, file_name = batch
            H, W = img1.shape[2], img1.shape[3]
            img1, img2 = img1.to(device), img2.to(device)

            if modal != 'CT':
                img2 = rgb2ycbcr(img2)
                img2_cbcr = img2[:, 1:3, :, :]
                img2 = img2[:, 0:1, :, :]  # Y 通道

            # 前向传播
            AS_F, feature1 = encoder(img1)
            BS_F, feature2 = encoder(img2)
            _, _, _, _, feature1, feature2, AU_F, BU_F = transfer(feature1, feature2)
            feature1 = project(feature1, [H, W]).to(device)
            feature2 = project(feature2, [H, W]).to(device)
            AU_F = project(AU_F, [H, W])
            BU_F = project(BU_F, [H, W])
            _, _, _, flow, _, _, W_map, (H1, H2, dH), R_feat = reg_net(feature1, feature2,img1, img2, raw1=AU_F, raw2=BU_F,      # <-- 你 pre-adapter 需要的输入
                                                            return_maps=True,
                                                            return_pre=True)
            warped_image2 = image_warp(flow, img2)
            fusion_y = fusion_net(AS_F, BS_F, AU_F, BU_F, flow)

            if modal != 'CT':
                fusion_cbcr = warp2D()(img2_cbcr, flow)
                fusion_rgb = torch.cat((fusion_y, fusion_cbcr), dim=1)
                fusion_img_to_save = ycbcr2rgb(fusion_rgb)
            else:
                fusion_img_to_save = fusion_y

            # 保存图像
            save_image(img1, os.path.join(figure_save_path, f"MRI/{file_name[0]}"))
            save_image(img2, os.path.join(figure_save_path, f"{modal}/{file_name[0]}"))
            save_image(fusion_img_to_save, os.path.join(figure_save_path, f"Fusion/{file_name[0]}"))
            save_image(warped_image2, os.path.join(figure_save_path, f"{modal}_align/{file_name[0]}"))

            save_image(W_map, os.path.join(figure_save_path, f"{modal}_W/{file_name[0]}"))
            save_image(norm01(H1), os.path.join(figure_save_path, f"{modal}_H1/{file_name[0]}"))
            save_image(norm01(H2), os.path.join(figure_save_path, f"{modal}_H2/{file_name[0]}"))
            save_image(norm01(dH), os.path.join(figure_save_path, f"{modal}_dH/{file_name[0]}"))

            # R_feat 可能是 [B,C,h,w]：先压成 1 通道，再上采样到 (H,W) 方便和 align 对齐看
            if R_feat.shape[1] > 1:
                R_vis = R_feat.mean(dim=1, keepdim=True)
            else:
                R_vis = R_feat

            R_vis = F.interpolate(R_vis, size=(H, W), mode="bilinear", align_corners=False)
            save_image(norm01(R_vis), os.path.join(figure_save_path, f"{modal}_R_feat/{file_name[0]}"))

            # 计算指标
            metrics = compute_all_metrics(fusion_y, img1, img2)
            metrics["filename"] = file_name[0]
            metric_records.append(metrics)

    # 计算平均值
    mean_metrics = {k: np.mean([m[k] for m in metric_records]) for k in ["QABF", "QSSIM", "QCV", "QVIF", "QS"]}
    mean_metrics["filename"] = "AVERAGE"

    # 保存CSV
    csv_path = os.path.join(figure_save_path, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "QABF", "QSSIM", "QCV", "QVIF", "QS"])
        writer.writeheader()
        writer.writerows(metric_records)
        writer.writerow(mean_metrics)

    print(f"\n[Metrics] {modal} 平均结果：")
    for k, v in mean_metrics.items():
        if k != "filename":
            print(f"   {k}: {v:.6f}")
    print(f"[Metrics] 已保存到 {csv_path}")

    return mean_metrics


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    warnings.filterwarnings('ignore')
    device = torch.device('cpu')
    modal = 'PET'   # 可改为 PET/SPECT
    checkpoint_path = './checkpoint'

    checkpoint = torch.load(os.path.join(checkpoint_path, f'EDARFusion-{modal}.pkl'),
                            map_location=torch.device('cpu'))

    encoder = Encoder().to(device)
    transfer = ModelTransfer_lite(num_vit=2, num_heads=4, img_size=[256, 256]).to(device)
    reg_net = EntropyDiffRegNet_lite(
        entropy_bins=16, entropy_win=9, entropy_sigma=0.08,
        tau=0.25, w_beta=1.0
    ).to(device)

    fusion_net = FusionNet_lite().to(device)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    transfer.load_state_dict(checkpoint['transfer_state_dict'])
    reg_net.load_state_dict(checkpoint['reg_net_state_dict'])
    fusion_net.load_state_dict(checkpoint['fusion_net_state_dict'])

    if modal == 'SPECT':
        val_dataset = TestData(
            img1_folder=f'./data/testData/{modal}/MRI',
            img2_folder=f'./data/testData/{modal}/{modal}',
            modal=modal
        )
    else:
        val_dataset = TestData(
            img1_folder=f'./data/testData/{modal}/MRI',
            img2_folder=f'./data/testData/{modal}/{modal}_RGB',
            modal=modal
        )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        num_workers=0
    )

    avg_metrics = validate_mask(encoder, transfer, reg_net, fusion_net, val_dataloader, modal)
