import torch
import torch.nn as nn
import numpy as np


class OffAxisPhysicsModule(nn.Module):
    """
    Physics Forward Model for dual off-axis holography.
    Real-valued implementation (no complex tensors) for full AMP/FP16 compatibility.

    Mathematical identity:
        |exp(jφ) + exp(jφ_ref)|^2 = 2 + 2*(cosφ * cosφ_ref + sinφ * sinφ_ref)

    Parameters
    ----------
    patch_size : int
        Image size (square).
    f1 : tuple(float, float)
        (fx1, fy1) — spatial frequency of reference beam 1, in cycles/pixel.
    f2 : tuple(float, float)
        (fx2, fy2) — spatial frequency of reference beam 2, in cycles/pixel.

    Conversion from physical parameters (for reference):
        fx = pixel_size * sin(θ_x) / λ
        fy = pixel_size * sin(θ_y) / λ
    """

    def __init__(
        self,
        patch_size: int = 256,
        f1: tuple = (0.190, 0.190),   # cycles/pixel  (~2°, λ=0.6328µm, px=3.45µm)
        f2: tuple = (0.475, 0.475),   # cycles/pixel  (~5°)
    ):
        super().__init__()

        self.patch_size = patch_size

        # Store frequencies as buffers (saved in checkpoint, visible in logs)
        self.register_buffer("f1", torch.tensor(f1, dtype=torch.float32))  # [fx, fy]
        self.register_buffer("f2", torch.tensor(f2, dtype=torch.float32))

        # Coordinate grid: pixel-index coords centred at 0
        x = np.arange(patch_size, dtype=np.float32) - patch_size / 2.0
        y = np.arange(patch_size, dtype=np.float32) - patch_size / 2.0
        XX, YY = np.meshgrid(x, y)   # [H, W]

        # ------------------------------------------------------------------
        # Precompute cos/sin of reference phase ONCE (no complex tensors)
        # ------------------------------------------------------------------
        def _ref_cos_sin(fx, fy):
            phase = (2.0 * np.pi * (fx * XX + fy * YY)).astype(np.float32)
            return torch.from_numpy(np.cos(phase)), torch.from_numpy(np.sin(phase))

        cos_r1, sin_r1 = _ref_cos_sin(*f1)
        cos_r2, sin_r2 = _ref_cos_sin(*f2)

        self.register_buffer("cos_ref1", cos_r1)   # [H, W]
        self.register_buffer("sin_ref1", sin_r1)
        self.register_buffer("cos_ref2", cos_r2)
        self.register_buffer("sin_ref2", sin_r2)

    def forward(self, pred_sc: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred_sc : [N, 2, H, W]
            channel 0 : sin(φ)
            channel 1 : cos(φ)

        Returns
        -------
        [N, 2, H, W]  — simulated holograms (H1, H2)
        """
        sin_phi = pred_sc[:, 0:1]   # [N, 1, H, W]
        cos_phi = pred_sc[:, 1:2]

        # Enforce unit circle: sin² + cos² = 1
        norm = torch.sqrt(cos_phi ** 2 + sin_phi ** 2 + 1e-8)
        cos_phi = cos_phi / norm
        sin_phi = sin_phi / norm

        # |O + R|^2 = 2 + 2*(cosφ*cosφ_ref + sinφ*sinφ_ref)
        H1 = 2.0 + 2.0 * (cos_phi * self.cos_ref1 + sin_phi * self.sin_ref1)
        H2 = 2.0 + 2.0 * (cos_phi * self.cos_ref2 + sin_phi * self.sin_ref2)

        return torch.cat([H1, H2], dim=1)   # [N, 2, H, W]