import torch
import torch.nn as nn
import numpy as np


class OffAxisPhysicsModule(nn.Module):
    """
    Physics Forward Model for dual off-axis holography.
    Real-valued implementation (no complex tensors) for full AMP/FP16 compatibility.

    Mathematical identity used:
        |exp(j*phi) + exp(j*phi_ref)|^2
            = 2 + 2*cos(phi - phi_ref)
            = 2 + 2*(cos(phi)*cos(phi_ref) + sin(phi)*sin(phi_ref))

    Since the model outputs [sin(phi), cos(phi)] and phi_ref is fixed,
    we precompute cos(phi_ref) and sin(phi_ref) once in __init__ as buffers.
    All operations remain real-valued → fully AMP/FP16 compatible → maximum speedup.
    """

    def __init__(
        self,
        patch_size=256,
        pixel_size=3.45,       # um
        wavelength=0.6328,     # um
        theta1_x=2.0,
        theta1_y=2.0,
        theta2_x=3.0,
        theta2_y=3.0,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.pixel_size = pixel_size
        self.wl = wavelength

        # Coordinate grid (pixel coords, centred)
        x = np.arange(patch_size, dtype=np.float32) - patch_size / 2.0
        y = np.arange(patch_size, dtype=np.float32) - patch_size / 2.0
        XX, YY = np.meshgrid(x, y)   # [H, W]

        # Store theta as buffers (for logging via get_frequencies)
        self.register_buffer("theta1", torch.tensor([theta1_x, theta1_y], dtype=torch.float32))
        self.register_buffer("theta2", torch.tensor([theta2_x, theta2_y], dtype=torch.float32))

        # ------------------------------------------------------------------
        # Precompute real-valued cos/sin of reference phase — done ONCE
        # Avoids complex tensors entirely → fully AMP-compatible
        # ------------------------------------------------------------------
        def _ref_cos_sin(theta_x_deg, theta_y_deg):
            fx = pixel_size * np.sin(np.deg2rad(theta_x_deg)) / wavelength
            fy = pixel_size * np.sin(np.deg2rad(theta_y_deg)) / wavelength
            phase = (2.0 * np.pi * (fx * XX + fy * YY)).astype(np.float32)  # [H, W]
            return torch.from_numpy(np.cos(phase)), torch.from_numpy(np.sin(phase))

        cos_r1, sin_r1 = _ref_cos_sin(theta1_x, theta1_y)
        cos_r2, sin_r2 = _ref_cos_sin(theta2_x, theta2_y)

        self.register_buffer("cos_ref1", cos_r1)   # [H, W]
        self.register_buffer("sin_ref1", sin_r1)   # [H, W]
        self.register_buffer("cos_ref2", cos_r2)   # [H, W]
        self.register_buffer("sin_ref2", sin_r2)   # [H, W]

    def get_frequencies(self):
        """Return spatial frequencies for logging."""
        theta1_rad = torch.deg2rad(self.theta1)
        theta2_rad = torch.deg2rad(self.theta2)
        fx1 = self.pixel_size * torch.sin(theta1_rad[0]) / self.wl
        fy1 = self.pixel_size * torch.sin(theta1_rad[1]) / self.wl
        fx2 = self.pixel_size * torch.sin(theta2_rad[0]) / self.wl
        fy2 = self.pixel_size * torch.sin(theta2_rad[1]) / self.wl
        return fx1, fy1, fx2, fy2

    def forward(self, pred_sc: torch.Tensor) -> torch.Tensor:
        """
        pred_sc : [N, 2, H, W]
            channel 0 : sin(phi)
            channel 1 : cos(phi)

        Returns simulated holograms : [N, 2, H, W]

        Formula (pure real ops, AMP/FP16 friendly):
            H = 2 + 2*(cos_phi * cos_ref + sin_phi * sin_ref)
        """
        sin_phi = pred_sc[:, 0:1]   # [N, 1, H, W]
        cos_phi = pred_sc[:, 1:2]   # [N, 1, H, W]

        # Enforce unit normalization
        norm = torch.sqrt(cos_phi ** 2 + sin_phi ** 2 + 1e-8)
        cos_phi = cos_phi / norm
        sin_phi = sin_phi / norm

        # Real-valued hologram: |O + R|^2 = 2 + 2*(cos_phi*cos_ref + sin_phi*sin_ref)
        # self.cos_ref1/sin_ref1 shape [H, W] → broadcast over [N, 1, H, W]
        H1 = 2.0 + 2.0 * (cos_phi * self.cos_ref1 + sin_phi * self.sin_ref1)  # [N, 1, H, W]
        H2 = 2.0 + 2.0 * (cos_phi * self.cos_ref2 + sin_phi * self.sin_ref2)  # [N, 1, H, W]

        return torch.cat([H1, H2], dim=1)   # [N, 2, H, W]