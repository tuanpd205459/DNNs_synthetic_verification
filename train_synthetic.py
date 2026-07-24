###############################################################
#  Synthetic Verification Training Script (Colab Fast Training)
#  Self-Supervised Multi-View Off-Axis Phase Reconstruction
#  - Learnable Reference Wavevectors (f1, f2)
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


def remove_linear_tilt(phase_map):
    """
    Removes 2D linear plane / residual reference carrier tilt (a*X + b*Y + c) from phase map.
    """
    H, W = phase_map.shape
    x = np.linspace(-1, 1, W, dtype=np.float32)
    y = np.linspace(-1, 1, H, dtype=np.float32)
    XX, YY = np.meshgrid(x, y)
    A = np.column_stack([XX.ravel(), YY.ravel(), np.ones(H * W, dtype=np.float32)])
    coeffs, _, _, _ = np.linalg.lstsq(A, phase_map.ravel(), rcond=None)
    plane = (A @ coeffs).reshape(H, W)
    return phase_map - plane


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

    batch_size = 4
    epochs = 100
    learning_rate = 3e-4
    patch_size = 256

    train_dataset = SyntheticHoloDataset(train_dir)
    val_dataset = SyntheticHoloDataset(val_dir) if os.path.exists(val_dir) else None
    test_dataset = SyntheticHoloDataset(test_dir)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    ) if val_dataset is not None else None

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    val_count = len(val_dataset) if val_dataset is not None else 0
    print(f"Loaded {len(train_dataset)} train samples, {val_count} val samples, {len(test_dataset)} test samples.")
    print(f"Config: batch_size={batch_size}, epochs={epochs}, lr={learning_rate}")

    model = PhaseUNet(in_channels=2, out_channels=2).to(device)
    physics_layer = OffAxisPhysicsModule(
        patch_size=patch_size,
        pixel_size=3.45,
        wavelength=0.6328,
        init_theta1_x=2.0,
        init_theta1_y=2.0,
        init_theta2_x=1.5,
        init_theta2_y=1.5,
        learnable_angles=True
    ).to(device)

    # Register both network parameters AND learnable physics layer parameters in Adam optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(physics_layer.parameters()),
        lr=learning_rate,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    maeloss = nn.L1Loss()
    os.makedirs('checkpoints', exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        physics_layer.train()
        t1 = default_timer()
        total_physics_loss = 0.0

        for xx, gt_phase, _, _ in train_loader:
            xx = xx.to(device)  # [N, 2, H, W]
            xx_norm = xx / torch.mean(xx, dim=(2, 3), keepdim=True)

            pred_sc = model(xx_norm)
            
            # Forward physics with self-calibrating learnable reference wavevectors
            im_x = physics_layer(pred_sc)

            loss = maeloss(im_x, xx_norm)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(physics_layer.parameters()), 1.0)
            optimizer.step()

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
                for idx, (xx, gt_phase, _, _) in enumerate(test_loader):
                    if idx >= eval_count:
                        break
                    xx = xx.to(device)
                    gt = gt_phase.squeeze().cpu().numpy()
                    gt_wrapped = np.angle(np.exp(1j * gt))

                    xx_norm = xx / torch.mean(xx, dim=(2, 3), keepdim=True)
                    pred_sc = model(xx_norm)
                    pred_ph = torch.atan2(pred_sc[:, 0:1, :, :], pred_sc[:, 1:2, :, :]).squeeze().cpu().numpy()

                    # 2D Linear Tilt Subtraction for fair phase comparison across random sample angles
                    pred_ph_c = remove_linear_tilt(pred_ph)
                    gt_wrapped_c = remove_linear_tilt(gt_wrapped)

                    data_range = gt_wrapped_c.max() - gt_wrapped_c.min()
                    val_mse += np.mean((pred_ph_c - gt_wrapped_c) ** 2)
                    val_ssim += ssim(gt_wrapped_c, pred_ph_c, data_range=data_range)
            val_mse /= eval_count
            val_ssim /= eval_count

            theta1, theta2 = physics_layer.get_angles()

            theta1 = theta1.detach().cpu().numpy()
            theta2 = theta2.detach().cpu().numpy()

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