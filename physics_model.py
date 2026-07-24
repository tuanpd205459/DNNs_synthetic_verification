import torch
import torch.nn as nn
import numpy as np


class OffAxisPhysicsModule(nn.Module):
    """
    Physics Forward Model for dual off-axis holography.

    Fixed parameters:
        theta1 = [theta1_x, theta1_y] (deg)
        theta2 = [theta2_x, theta2_y] (deg)

    Reference wave:
        fx = pixel_size * sin(theta_x) / wavelength
        fy = pixel_size * sin(theta_y) / wavelength

        R = exp(j*2*pi*(fx*x + fy*y))

    Forward model:
        I = |O + R|^2
    """

    def __init__(
        self,
        patch_size=256,
        pixel_size=3.45,          # um
        wavelength=0.6328,        # um
        theta1_x=2.0,
        theta1_y=2.0,
        theta2_x=3.0,
        theta2_y=3.0,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.pixel_size = pixel_size
        self.wl = wavelength

        # ------------------------------------------------------------
        # Coordinate grid (pixel coordinates)
        # ------------------------------------------------------------
        x = np.arange(patch_size, dtype=np.float32) - patch_size / 2.0
        y = np.arange(patch_size, dtype=np.float32) - patch_size / 2.0
        XX, YY = np.meshgrid(x, y)

        self.register_buffer("XX", torch.from_numpy(XX))
        self.register_buffer("YY", torch.from_numpy(YY))

        # ------------------------------------------------------------
        # Khởi tạo góc cố định (không dùng nn.Parameter)
        # ------------------------------------------------------------
        theta1 = torch.tensor([theta1_x, theta1_y], dtype=torch.float32)
        theta2 = torch.tensor([theta2_x, theta2_y], dtype=torch.float32)

        # Đăng ký dưới dạng buffer để model tự quản lý device (CPU/GPU) 
        # nhưng không tính gradient (không learnable)
        self.register_buffer("theta1", theta1)
        self.register_buffer("theta2", theta2)

    # -----------------------------------------------------------------
    # Convert angle -> spatial frequency
    # -----------------------------------------------------------------
    def get_frequencies(self):
        # Chuyển đổi trực tiếp từ buffer độ sang radian
        theta1_rad = torch.deg2rad(self.theta1)
        theta2_rad = torch.deg2rad(self.theta2)

        fx1 = self.pixel_size * torch.sin(theta1_rad[0]) / self.wl
        fy1 = self.pixel_size * torch.sin(theta1_rad[1]) / self.wl

        fx2 = self.pixel_size * torch.sin(theta2_rad[0]) / self.wl
        fy2 = self.pixel_size * torch.sin(theta2_rad[1]) / self.wl

        return fx1, fy1, fx2, fy2

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(self, pred_sc):
        """
        pred_sc: [N, 2, H, W]

            channel 0 : sin(phi)
            channel 1 : cos(phi)

        return:
            simulated holograms
            [N,2,H,W]
        """

        sin_phi = pred_sc[:, 0:1]
        cos_phi = pred_sc[:, 1:2]

        # Complex object field
        object_field = (cos_phi + 1j * sin_phi).squeeze(1)

        # Fixed reference frequencies
        fx1, fy1, fx2, fy2 = self.get_frequencies()

        # Reference waves
        ref1 = torch.exp(
            1j * 2.0 * np.pi *
            (fx1 * self.XX + fy1 * self.YY)
        )

        ref2 = torch.exp(
            1j * 2.0 * np.pi *
            (fx2 * self.XX + fy2 * self.YY)
        )

        # Simulated holograms
        H1 = torch.abs(object_field + ref1) ** 2
        H2 = torch.abs(object_field + ref2) ** 2

        return torch.stack([H1, H2], dim=1)