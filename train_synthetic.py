###############################################################
#  Synthetic Verification Training Script (Colab Fast Training)
#  Self-Supervised Multi-View Off-Axis Phase Reconstruction
#  - Fixed Reference Wavevectors (f1, f2)
#  - Minimal Hologram Matching Loss (Loss 1)
#  - Continuous [sin(phi), cos(phi)] Unit-Normalized Representation
###############################################################

import os
import torch
import torch.nn as nn
import numpy as np
from timeit import default_timer
from skimage.metrics import structural_similarity as ssim

from physics_model import OffAxisPhysicsModule
from model import PhaseUNet
from dataset import SyntheticHoloDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tv_loss(inputs):
    # inputs: [N, C, H, W]
    n, c, h, w = inputs.shape
    grad_x = inputs[:,:,1:,:] - inputs[:,:,:-1,:]
    grad_y = inputs[:,:,:,1:] - inputs[:,:,:,:-1]
    tv = (grad_x.abs().sum() + grad_y.abs().sum()) / (n*c*h*w)
    return tv


def main():
    print(f"======================================================")
    print(f" Starting Verification Training on Device: [{device.type.upper()}]")
    if device.type == 'cpu':
        print(" ⚠️ WARNING: Code is running on CPU! Enable GPU in Colab (Runtime -> Change runtime type -> T4 GPU) to run 50x faster!")
    else:
        print(f" 🚀 GPU Active: {torch.cuda.get_device_name(0)}")
    print(f"======================================================")

    train_dir = 'data_synth/train'
    val_dir = 'data_synth/val'
    test_dir = 'data_synth/test'

    # If dataset missing or empty, generate synthetic samples
    if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
        print("[Notice] Generating synthetic dataset with random reference angles...")
        import generate_synthetic_data
        generate_synthetic_data.main()

    batch_size = 16   # tăng từ 4→16: giảm 4× số batches/epoch
    epochs = 100
    learning_rate = 3e-4

    train_dataset = SyntheticHoloDataset(
        train_dir,
        training=True
    )

    test_dataset = SyntheticHoloDataset(
        test_dir,
        training=False
    )
    val_dataset = SyntheticHoloDataset(val_dir) if os.path.exists(val_dir) else None
    
    # ---------------------------------------------------------
    # Tự động trích xuất patch_size từ tập dữ liệu thay vì hardcode
    # ---------------------------------------------------------
    sample_data = train_dataset[0]
    # Kiểm tra xem dataset trả về tuple hay tensor trực tiếp
    if isinstance(sample_data, tuple) or isinstance(sample_data, list):
        patch_size = sample_data[0].shape[-1]
    else:
        patch_size = sample_data.shape[-1]
    print(f"Auto-detected patch_size from dataset: {patch_size}")

    num_workers = min(8, os.cpu_count())

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    ) if val_dataset is not None else None

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    val_count = len(val_dataset) if val_dataset is not None else 0
    print(f"Loaded {len(train_dataset)} train samples, {val_count} val samples, {len(test_dataset)} test samples.")
    print(f"Config: batch_size={batch_size}, epochs={epochs}, lr={learning_rate}")

    model = PhaseUNet(in_channels=2, out_channels=2).to(device)
    
    # 1. Khởi tạo physics_layer với các góc cố định và patch_size động
    physics_layer = OffAxisPhysicsModule(
        patch_size=patch_size,
        pixel_size=3.45,
        wavelength=0.6328,
        theta1_x=2.0,
        theta1_y=2.0,
        theta2_x=3.0,
        theta2_y=3.0
    ).to(device)

    # torch.compile: fuses kernels, tối ưu graph → tăng thêm ~20–30% tốc độ
    # (PyTorch 2.0+, chỉ dùng khi có GPU)
    if torch.cuda.is_available() and hasattr(torch, 'compile'):
        model = torch.compile(model, mode='reduce-overhead')
        print("✅ torch.compile enabled (mode=reduce-overhead)")

    # 2. Chỉ đưa các tham số của mạng UNet vào Optimizer (physics_layer không learnable)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    maeloss = nn.L1Loss()
    os.makedirs('checkpoints', exist_ok=True)

    # AMP: Automatic Mixed Precision — tận dụng Tensor Cores của T4 (FP16)
    # Thường tăng tốc 2–3× với gần như không giảm accuracy
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for ep in range(1, epochs + 1):
        model.train()
        t1 = default_timer()
        total_physics_loss = 0.0

        for batch in train_loader:
            # Xử lý an toàn trường hợp train_loader trả về tuple thay vì tensor đơn lẻ
            if isinstance(batch, tuple) or isinstance(batch, list):
                xx = batch[0]
            else:
                xx = batch

            xx = xx.to(device, non_blocking=True)

            optimizer.zero_grad()

            # AMP autocast: tự động dùng FP16 cho các op phù hợp
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred_sc = model(xx)
                im_x = physics_layer(pred_sc)
                loss = maeloss(im_x, xx)

            # Scaled backward + optimizer step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_physics_loss += loss.item()

        scheduler.step()
        t2 = default_timer()

        avg_loss = total_physics_loss / len(train_loader)
        print(f"Epoch [{ep:03d}/{epochs}] Time: {t2-t1:.2f}s | Loss: {avg_loss:.4f}", end='')

        # Fast evaluation on up to 5 test samples every 10 epochs
        if ep % 10 == 0 or ep == epochs or ep == 1:
            model.eval()
            val_mse = 0.0
            val_ssim = 0.0
            eval_count = min(5, len(test_loader))

            with torch.no_grad():
                for idx, (xx, gt_phase, sample_name) in enumerate(test_loader):
                    if idx >= eval_count:
                        break
                    xx = xx.to(device)
                    gt = gt_phase.squeeze().cpu().numpy()
                    gt_wrapped = np.angle(np.exp(1j * gt))

                    # Truyền trực tiếp xx vào model
                    pred_sc = model(xx)
                    pred_ph = torch.atan2(pred_sc[:, 0:1, :, :], pred_sc[:, 1:2, :, :]).squeeze().cpu().numpy()

                    data_range = gt_wrapped.max() - gt_wrapped.min()
                    val_ssim += ssim(gt_wrapped, pred_ph, data_range=data_range)
                    diff = np.angle(np.exp(1j * (pred_ph - gt_wrapped)))
                    val_mse += np.mean(diff ** 2)
            val_mse /= eval_count
            val_ssim /= eval_count

            # Lấy trực tiếp tham số theta từ buffer của physics_layer
            theta1 = physics_layer.theta1.cpu().numpy()
            theta2 = physics_layer.theta2.cpu().numpy()

            fx1, fy1, fx2, fy2 = physics_layer.get_frequencies()

            fx1 = fx1.item()
            fy1 = fy1.item()
            fx2 = fx2.item()
            fy2 = fy2.item()

            print(
                f" | SSIM: {val_ssim:.4f}"
                f" | MSE: {val_mse:.6f}"
                f" | Theta1=({theta1[0]:.3f}°, {theta1[1]:.3f}°)"
                f" | Theta2=({theta2[0]:.3f}°, {theta2[1]:.3f}°)"
                f" | f1=({fx1:.4f}, {fy1:.4f})"
                f" | f2=({fx2:.4f}, {fy2:.4f})",
                end=''
            )
            
            torch.save({
                'model': model.state_dict(),
                'physics_layer': physics_layer.state_dict(), 
                'epoch': ep
            }, 'checkpoints/best_synthetic_model.pth')

        print()

    print("Verification Training Completed!")


if __name__ == '__main__':
    main()