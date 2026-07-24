import os
import glob
import torch
import numpy as np
import PIL.Image


class SyntheticHoloDataset(torch.utils.data.Dataset):
    """
    Dataset loader for synthetic holography.

    Training mode:
        Input :
            H1, H2
        Return:
            input_hologram

    Evaluation mode:
        Input :
            H1, H2
            Ground-truth phase
        Return:
            input_hologram,
            gt_phase,
            sample_name
    """

    def __init__(self, data_dir, training=True):
        self.data_dir = data_dir
        self.training = training
        self.samples = sorted(glob.glob(os.path.join(data_dir, "sample_*")))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample_path = self.samples[index]

        h1_file = os.path.join(sample_path, "hologram1.npy")
        h2_file = os.path.join(sample_path, "hologram2.npy")

        if os.path.exists(h1_file):

            H1 = np.load(h1_file).astype(np.float32)
            H2 = np.load(h2_file).astype(np.float32)

            if not self.training:
                gt_phase = np.load(
                    os.path.join(sample_path, "gt_phase.npy")
                ).astype(np.float32)

        else:

            H1 = (
                np.array(
                    PIL.Image.open(os.path.join(sample_path, "hologram1.png"))
                ).astype(np.float32)
                / 255.0
            )

            H2 = (
                np.array(
                    PIL.Image.open(os.path.join(sample_path, "hologram2.png"))
                ).astype(np.float32)
                / 255.0
            )

            if not self.training:
                gt_phase = (
                    np.array(
                        PIL.Image.open(os.path.join(sample_path, "gt_phase.png"))
                    ).astype(np.float32)
                    / 255.0
                    * (10.0 * np.pi)
                )

        # Mean intensity normalization
        H1 /= (H1.mean() + 1e-8)
        H2 /= (H2.mean() + 1e-8)

        inp = np.stack([H1, H2], axis=0)

        if self.training:
            return torch.from_numpy(inp)

        gt = gt_phase[np.newaxis, ...]
        sample_name = os.path.basename(sample_path)

        return (
            torch.from_numpy(inp),
            torch.from_numpy(gt),
            sample_name,
        )