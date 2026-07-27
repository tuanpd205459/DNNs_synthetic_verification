import os
import math
import numpy as np
from pathlib import Path
import PIL.Image
from tqdm import tqdm
# Optional: if you have scipy installed, you can use a more efficient Zernike implementation.
# from scipy.special import factorial
def zernike_eval(n: int, m: int, rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Evaluate Zernike polynomial Z_n^m(rho, phi).
    Parameters
    ----------
    n, m : int
        Radial and azimuthal orders (|m| <= n, n‑|m| even).
    rho, phi : np.ndarray
        Polar coordinates normalized to the unit circle.
    Returns
    -------
    np.ndarray
        The Zernike mode evaluated on the supplied grid.
    """
    if (n - abs(m)) % 2 != 0:
        return np.zeros_like(rho)
    # Vectorised computation of the radial polynomial R_n^m
    k = np.arange((n - abs(m)) // 2 + 1)
    # Using math.factorial for integer factorials – fast enough for n <= 5
    c = ((-1) ** k *
         np.vectorize(math.factorial)(n - k) /
         (np.vectorize(math.factorial)(k) *
          np.vectorize(math.factorial)((n + abs(m)) // 2 - k) *
          np.vectorise(math.factorial)((n - abs(m)) // 2 - k)))
    # Broadcast c over the rho grid
    R = (c[:, None, None] * rho ** (n - 2 * k[:, None, None])).sum(axis=0)
    if m >= 0:
        return R * np.cos(m * phi)
    else:
        return R * np.sin(-m * phi)
def generate_zernike_phase_map(
        shape: tuple[int, int] = (256, 256),
        max_phase: float = 10 * np.pi,
        min_active_modes: int = 2,
        max_active_modes: int = 8) -> np.ndarray:
    """Create a random phase map built from a subset of Zernike modes.
    Parameters
    ----------
    shape : tuple[int, int]
        Output image size (height, width).
    max_phase : float
        Maximum phase value after normalisation (radians).
    min_active_modes, max_active_modes : int
        Range of how many Zernike modes are randomly activated.
    Returns
    -------
    np.ndarray (float32)
        Normalised phase map with values in ``[0, max_phase]``.
    """
    H, W = shape
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    XX, YY = np.meshgrid(x, y)
    rho = np.sqrt(XX ** 2 + YY ** 2)
    phi = np.arctan2(YY, XX)
    # First 21 Zernike modes (n <= 5)
    z_modes = [
        (0, 0),
        (1, -1), (1, 1),
        (2, -2), (2, 0), (2, 2),
        (3, -3), (3, -1), (3, 1), (3, 3),
        (4, -4), (4, -2), (4, 0), (4, 2), (4, 4),
        (5, -5), (5, -3), (5, -1),
        (5, 1), (5, 3), (5, 5)
    ]
    rng = np.random.default_rng()
    n_active = rng.integers(min_active_modes, max_active_modes + 1)
    # Guard against requesting more active modes than available
    n_active = min(n_active, len(z_modes))
    active_idx = rng.choice(len(z_modes), size=n_active, replace=False)
    coeffs = np.zeros(len(z_modes))
    coeffs[active_idx] = rng.uniform(-5.0, 5.0, n_active)
    phase = np.zeros_like(rho)
    for coeff, (n, m) in zip(coeffs, z_modes):
        if coeff != 0:
            phase += coeff * zernike_eval(n, m, rho, phi)
    # Normalise to [0, max_phase]
    phase -= phase.min()
    if phase.max() > 0:
        phase /= phase.max()
    phase *= max_phase
    return phase.astype(np.float32)
def simulate_offaxis_holograms(
        phase_map: np.ndarray,
        wavelength: float = 0.6328,
        pixel_size: float = 3.45,
        theta1: tuple[float, float] = (2.0, 2.0),
        theta2: tuple[float, float] = (3.0, 3.0)) -> tuple[np.ndarray, np.ndarray]:
    """Simulate two off‑axis holograms with fixed reference beam angles.
    Parameters
    ----------
    phase_map : np.ndarray
        Phase map (radians) to be encoded.
    wavelength : float
        Illumination wavelength (micrometres).
    pixel_size : float
        Sensor pixel size (micrometres).
    theta1, theta2 : tuple[float, float]
        Angles (degrees) for the two reference beams (theta_x, theta_y).
    Returns
    -------
    (H1, H2) : tuple[np.ndarray, np.ndarray]
        Intensity holograms for the two reference angles.
    """
    H, W = phase_map.shape
    # Use ogrid to avoid allocating a full meshgrid twice.
    y = np.arange(H, dtype=np.float32) - H / 2.0
    x = np.arange(W, dtype=np.float32) - W / 2.0
    YY, XX = np.ogrid[:H, :W]
    def _reference_beam(theta):
        thx, thy = theta
        fx = pixel_size * np.sin(np.deg2rad(thx)) / wavelength
        fy = pixel_size * np.sin(np.deg2rad(thy)) / wavelength
        return np.exp(1j * 2 * np.pi * (fx * XX + fy * YY))
    R1 = _reference_beam(theta1)
    R2 = _reference_beam(theta2)
    U = np.exp(1j * phase_map)
    H1 = np.abs(U + R1) ** 2
    H2 = np.abs(U + R2) ** 2
    return H1.astype(np.float32), H2.astype(np.float32)
def main() -> None:
    """Generate a synthetic dataset of phase maps and their off‑axis holograms.
    The data layout is::
        data_synth/
            train/   sample_00000/ ...
            val/     sample_00000/ ...
            test/    sample_00000/ ...
    Each ``sample_xxxxx`` folder contains:
        - ``gt_phase.npy``        : ground‑truth phase (float32)
        - ``hologram1.npy``       : hologram with reference angle 1
        - ``hologram2.npy``       : hologram with reference angle 2
        - ``gt_phase.png`` / ``hologram1.png`` / ``hologram2.png``
          for quick visual inspection (8‑bit PNG).
    """
    out_dir = Path("data_synth")
    splits = [
        ("train", 200, 0),
        ("val",   10, 100_000),
        ("test",  10, 200_000),
    ]
    for mode, _, _ in splits:
        (out_dir / mode).mkdir(parents=True, exist_ok=True)
    print("Generating synthetic dataset with FIXED reference beam angles...")
    rng = np.random.default_rng()
    for mode, count, seed_offset in splits:
        for idx in tqdm(range(count), desc=mode.capitalize()):
            # Seed is deterministic per sample for reproducibility
            rng.seed(seed_offset + idx)
            gt_phase = generate_zernike_phase_map(shape=(256, 256), max_phase=10 * np.pi)
            H1, H2 = simulate_offaxis_holograms(gt_phase)
            sample_dir = out_dir / mode / f"sample_{idx:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            # Save raw arrays – npz keeps them together and compresses them.
            np.savez_compressed(sample_dir / "data.npz",
                                gt_phase=gt_phase,
                                hologram1=H1,
                                hologram2=H2)
            # Also save PNGs for quick visual checks (optional).
            for arr, name in [(gt_phase, "gt_phase"), (H1, "hologram1"), (H2, "hologram2")]:
                img = (arr / arr.max() * 255).astype(np.uint8)
                PIL.Image.fromarray(img).save(sample_dir / f"{name}.png")
    print(f"✅ Generated synthetic dataset at '{out_dir}/'")
if __name__ == "__main__":
    main()