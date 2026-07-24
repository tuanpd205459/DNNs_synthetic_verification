import os
import glob
import torch
import numpy as np
import PIL.Image


class SyntheticHoloDataset(torch.utils.data.Dataset):
    """
    Dataset loader for synthetic phase verification experiments.
    Loads (H1, H2) as 2-channel input, gt_phase, and sample-specific reference frequencies (f1, f2).
    """
    def __init__(self, data_dir, patch_size=512):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.samples = sorted(glob.glob(os.path.join(data_dir, 'sample_*')))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]

        h1_file = os.path.join(sample_path, 'hologram1.npy')
        h2_file = os.path.join(sample_path, 'hologram2.npy')
        gt_file = os.path.join(sample_path, 'gt_phase.npy')
        freq_file = os.path.join(sample_path, 'freq_params.npy')

        if os.path.exists(h1_file):
            H1 = np.load(h1_file).astype('float32')
            H2 = np.load(h2_file).astype('float32')
            gt_phase = np.load(gt_file).astype('float32')
            freq_params = np.load(freq_file).astype('float32')  # [2, 2] -> f1, f2
        else:
            H1 = np.array(PIL.Image.open(os.path.join(sample_path, 'hologram1.png'))).astype('float32') / 255.0
            H2 = np.array(PIL.Image.open(os.path.join(sample_path, 'hologram2.png'))).astype('float32') / 255.0
            gt_phase = np.array(PIL.Image.open(os.path.join(sample_path, 'gt_phase.png'))).astype('float32') / 255.0 * (10.0 * np.pi)
            freq_params = np.zeros((2, 2), dtype=np.float32)

        # Mean intensity normalization for holograms
        H1 = H1 / (np.mean(H1) + 1e-8)
        H2 = H2 / (np.mean(H2) + 1e-8)

        inp = np.stack([H1, H2], axis=0)  # [2, H, W]
        gt = gt_phase[np.newaxis, ...]     # [1, H, W]

        sample_name = os.path.basename(sample_path)
        return torch.from_numpy(inp), torch.from_numpy(gt), torch.from_numpy(freq_params), sample_name
