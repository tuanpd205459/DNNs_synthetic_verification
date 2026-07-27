###############################################################
#  Synthetic Verification Testing & Visualization Script
#  Compares Ground-Truth Phase vs AI Reconstructed Phase
#  Evaluation on Unwrapped Phase
###############################################################

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import unwrap_phase

from model import PhaseUNet
from dataset import SyntheticHoloDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    print("Running Evaluation on Synthetic Test Dataset...")

    test_dir = "data_synth/test"

    if not os.path.exists(test_dir):
        print("[Error] Test dataset not found.")
        return

    # Sửa 1: Thêm cờ training=False để lấy ground truth
    test_dataset = SyntheticHoloDataset(
        test_dir,
        training=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    ckpt_path = "checkpoints/best_synthetic_model.pth"

    if not os.path.exists(ckpt_path):
        print("[Error] Checkpoint not found.")
        return

    model = PhaseUNet(
        in_channels=2,
        out_channels=2
    ).to(device)

    checkpoint = torch.load(
        ckpt_path,
        map_location=device
    )

    # Chỉ load model, không cần physics_layer khi inference
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output_dir = "results_synthetic"
    os.makedirs(output_dir, exist_ok=True)

    ssim_list = []
    mse_list = []

    with torch.no_grad():

        # Sửa 2: Chỉ unpack 3 biến theo đúng interface mới của Dataset
        for i, (xx, gt_phase, sample_name) in enumerate(test_loader):

            xx = xx.to(device)

            sample_id = sample_name

            ##################################################
            # Ground Truth
            ##################################################

            # Đảm bảo chuyển đổi về NumPy 2D array
            gt_unwrapped = gt_phase.squeeze().cpu().numpy()

            ##################################################
            # Prediction
            ##################################################

            pred_sc = model(xx)

            pred_wrapped = torch.atan2(
                pred_sc[:, 0:1],
                pred_sc[:, 1:2]
            ).squeeze().cpu().numpy()

            ##################################################
            # Phase Unwrapping
            ##################################################

            pred_unwrapped = unwrap_phase(pred_wrapped)

            ##################################################
            # Metrics
            ##################################################

            # Remove global phase offset after unwrapping
            offset = np.mean(gt_unwrapped - pred_unwrapped)
            pred_unwrapped = pred_unwrapped + offset

            data_range = gt_unwrapped.max() - gt_unwrapped.min()

            cur_ssim = ssim(
                gt_unwrapped,
                pred_unwrapped,
                data_range=data_range
            )

            cur_mse = np.mean(
                (gt_unwrapped - pred_unwrapped) ** 2
            )

            ssim_list.append(cur_ssim)
            mse_list.append(cur_mse)

            ##################################################
            # Visualization
            ##################################################

            fig, axes = plt.subplots(
                1,
                5,
                figsize=(22, 4)
            )

            H1 = xx[0, 0].cpu().numpy()

            im0 = axes[0].imshow(
                H1,
                cmap="gray"
            )
            axes[0].set_title("Input Hologram")
            axes[0].axis("off")

            # Sử dụng gt_unwrapped thay vì gt_phase
            im1 = axes[1].imshow(
                gt_unwrapped,
                cmap="viridis"
            )
            axes[1].set_title("GT Unwrapped")
            axes[1].axis("off")
            plt.colorbar(im1, ax=axes[1], fraction=0.046)

            im2 = axes[2].imshow(
                pred_unwrapped,
                cmap="viridis"
            )
            axes[2].set_title(
                f"AI Unwrapped\nSSIM={cur_ssim:.4f}"
            )
            axes[2].axis("off")
            plt.colorbar(im2, ax=axes[2], fraction=0.046)

            # Tính toán sai số giữa hai NumPy arrays
            diff = gt_unwrapped - pred_unwrapped
            
            # Cân bằng giá trị vmin, vmax để màu trắng đại diện cho 0
            vmax_diff = np.max(np.abs(diff))
            im3 = axes[3].imshow(
                diff,
                cmap="bwr",
                vmin=-vmax_diff,
                vmax=vmax_diff
            )
            axes[3].set_title("Difference")
            axes[3].axis("off")
            plt.colorbar(im3, ax=axes[3], fraction=0.046)

            err = np.abs(diff)

            im4 = axes[4].imshow(
                err,
                cmap="hot"
            )
            axes[4].set_title(
                f"Absolute Error\nMSE={cur_mse:.6f}"
            )
            axes[4].axis("off")
            plt.colorbar(im4, ax=axes[4], fraction=0.046)

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{sample_id}_verification.png"
                ),
                dpi=150
            )

            plt.close()

            print(
                f"[{i+1:03d}/{len(test_loader)}] "
                f"{sample_id} | "
                f"SSIM={cur_ssim:.4f} | "
                f"MSE={cur_mse:.6f}"
            )

    print("\n======================================================")

    print(f"Mean SSIM : {np.mean(ssim_list):.4f}")
    print(f"Mean MSE  : {np.mean(mse_list):.6f}")

    print(f"Results saved to: {output_dir}")

    print("======================================================")


if __name__ == "__main__":
    main()