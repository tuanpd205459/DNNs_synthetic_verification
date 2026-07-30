import os
import glob
import torch
import numpy as np


class SyntheticHoloDataset(torch.utils.data.Dataset):
    """
    Dataset loader for synthetic holography.

    Expects each sample folder to contain ``data.npz`` with keys:
        - ``hologram1``  : float32 [H, W]
        - ``hologram2``  : float32 [H, W]
        - ``gt_phase``   : float32 [H, W]   (only needed in eval mode)

    Training mode  → returns: input_tensor  [2, H, W]
    Eval mode      → returns: (input_tensor, gt_phase_tensor [1,H,W], sample_name)
    """

    def __init__(self, data_dir, training=True):
        self.data_dir = data_dir
        self.training = training
        self.samples = sorted(glob.glob(os.path.join(data_dir, "sample_*")))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No sample folders found in '{data_dir}'. "
                "Run generate_synthetic_data.py first."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]
        npz_path = os.path.join(sample_path, "data.npz")

        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Missing data.npz in '{sample_path}'. "
                "Re-generate the dataset with generate_synthetic_data.py."
            )

        data = np.load(npz_path)
        H1 = data["hologram1"].astype(np.float32)   # [H, W]
        H2 = data["hologram2"].astype(np.float32)   # [H, W]

        # Mean intensity normalization (stabilises training)
        # H1: [N,1,H,W]
        H1 = H1 / 4.0
        H2 = H2 / 4.0

        inp = np.stack([H1, H2], axis=0)             # [2, H, W]

        if self.training:
            return torch.from_numpy(inp)

        gt_phase = data["gt_phase"].astype(np.float32)  # [H, W]
        gt = gt_phase[np.newaxis, ...]                   # [1, H, W]
        sample_name = os.path.basename(sample_path)

        return (
            torch.from_numpy(inp),
            torch.from_numpy(gt),
            sample_name,
        )