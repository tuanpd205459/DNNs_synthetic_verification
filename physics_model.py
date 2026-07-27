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
        R  = exp(j * 2*pi * (fx*x + fy*y))

    Forward model:
        I = |O + R|^2

    Optimization:
        Reference waves are precomputed once in __init__ and stored as
        buffers — they are constant because theta is FIXED (non-learnable).
        This avoids recomputing exp(j*...) on every forward call, which was
        the main reason each epoch took 4–5 minutes.
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
        # Coordinate grid (pixel coordinates, centred at 0)
        # ------------------------------------------------------------
        x = np.arange(patch_size, dtype=np.float64) - patch_size / 2.0
        y = np.arange(patch_size, dtype=np.float64) - patch_size / 2.0
        XX, YY = np.meshgrid(x, y)          # [H, W]

        # Store theta as buffers so .to(device) works automatically
        theta1 = torch.tensor([theta1_x, theta1_y], dtype=torch.float32)
        theta2 = torch.tensor([theta2_x, theta2_y], dtype=torch.float32)
        self.register_buffer("theta1", theta1)
        self.register_buffer("theta2", theta2)

        # ------------------------------------------------------------
        # Precompute reference waves ONCE (theta is fixed → waves are constant)
        # Avoids repeating exp() on every forward() call → huge speedup
        # ------------------------------------------------------------
        def _make_ref(theta_x_deg, theta_y_deg):
            fx = pixel_size * np.sin(np.deg2rad(theta_x_deg)) / wavelength
            fy = pixel_size * np.sin(np.deg2rad(theta_y_deg)) / wavelength
            phase = 2.0 * np.pi * (fx * XX + fy * YY)          # [H, W]
            wave = np.exp(1j * phase).astype(np.complex64)      # [H, W]
            # Split into real/imag because PyTorch buffers support complex64
            return torch.from_numpy(wave)

        self.register_buffer("ref1", _make_ref(theta1_x, theta1_y))  # [H, W] complex64
        self.register_buffer("ref2", _make_ref(theta2_x, theta2_y))  # [H, W] complex64

    # -----------------------------------------------------------------
    # Convert stored angle buffers → spatial frequencies (for logging)
    # -----------------------------------------------------------------
    def get_frequencies(self):
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

        return: simulated holograms [N, 2, H, W]
        """
        sin_phi = pred_sc[:, 0:1]   # [N, 1, H, W]
        cos_phi = pred_sc[:, 1:2]   # [N, 1, H, W]

        # Enforce unit normalization on S^1 circle
        norm = torch.sqrt(cos_phi ** 2 + sin_phi ** 2 + 1e-8)
        cos_phi = cos_phi / norm
        sin_phi = sin_phi / norm

        # Complex object field [N, H, W]
        object_field = (cos_phi + 1j * sin_phi).squeeze(1)

        # Use precomputed reference waves (broadcast over batch dim)
        # ref1, ref2 shape: [H, W] → broadcasts to [N, H, W]
        H1 = torch.abs(object_field + self.ref1) ** 2   # [N, H, W]
        H2 = torch.abs(object_field + self.ref2) ** 2   # [N, H, W]

        return torch.stack([H1, H2], dim=1)              # [N, 2, H, W]