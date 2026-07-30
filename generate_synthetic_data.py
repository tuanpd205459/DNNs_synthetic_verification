import os
import numpy as np
from pathlib import Path
import PIL.Image
from tqdm import tqdm

def zernike_mode(j, rho, phi):
    """
    Zernike polynomials (Noll indexing) - Non-normalized
    j = 1 ... 21
    rho : normalized radius [0,1]
    phi : angle (rad)
    """
    Z = np.zeros_like(rho)

    if j == 1:
        Z = np.ones_like(rho)
    elif j == 2:
        Z = rho * np.cos(phi)
    elif j == 3:
        Z = rho * np.sin(phi)
    elif j == 4:
        Z = (2 * rho**2 - 1)
    elif j == 5:
        Z = rho**2 * np.sin(2 * phi)
    elif j == 6:
        Z = rho**2 * np.cos(2 * phi)
    elif j == 7:
        Z = (3 * rho**3 - 2 * rho) * np.sin(phi)
    elif j == 8:
        Z = (3 * rho**3 - 2 * rho) * np.cos(phi)
    elif j == 9:
        Z = rho**3 * np.sin(3 * phi)
    elif j == 10:
        Z = rho**3 * np.cos(3 * phi)
    elif j == 11:
        Z = (6 * rho**4 - 6 * rho**2 + 1)
    elif j == 12:
        Z = (4 * rho**4 - 3 * rho**2) * np.cos(2 * phi)
    elif j == 13:
        Z = (4 * rho**4 - 3 * rho**2) * np.sin(2 * phi)
    elif j == 14:
        Z = rho**4 * np.cos(4 * phi)
    elif j == 15:
        Z = rho**4 * np.sin(4 * phi)
    elif j == 16:
        Z = (10 * rho**5 - 12 * rho**3 + 3 * rho) * np.cos(phi)
    elif j == 17:
        Z = (10 * rho**5 - 12 * rho**3 + 3 * rho) * np.sin(phi)
    elif j == 18:
        Z = (5 * rho**5 - 4 * rho**3) * np.cos(3 * phi)
    elif j == 19:
        Z = (5 * rho**5 - 4 * rho**3) * np.sin(3 * phi)
    elif j == 20:
        Z = rho**5 * np.cos(5 * phi)
    elif j == 21:
        Z = rho**5 * np.sin(5 * phi)

    return Z


def generate_zernike_phase_map(
    shape=(256, 256),
    max_phase=4 * np.pi,
    min_modes=2,
    max_modes=8,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    H, W = shape

    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)

    X, Y = np.meshgrid(x, y)

    rho = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)

    mask = rho <= 1

    phase = np.zeros_like(rho)

    n_modes = rng.integers(min_modes, max_modes + 1)

    modes = rng.choice(
        np.arange(2, 22),
        size=n_modes,
        replace=False
    )

    coeff = rng.uniform(-1, 1, n_modes)

    for c, j in zip(coeff, modes):
        phase += c * zernike_mode(j, rho, phi)


    if phase.max() > 0:
        phase /= phase.max()

    phase *= max_phase

    return phase.astype(np.float32)


def simulate_offaxis_holograms(
    phase,
    wavelength=0.6328,
    pixel_size=3.45,
    theta1=(1.5, 1.5),
    theta2=(3.0, 3.0)
):
    H, W = phase.shape

    # Tối ưu hóa tính toán tọa độ bằng ogrid tương tự logic bạn đưa ra
    YY, XX = np.ogrid[:H, :W]
    YY = YY.astype(np.float32) - H / 2.0
    XX = XX.astype(np.float32) - W / 2.0

    x_phys = XX * pixel_size
    y_phys = YY * pixel_size

    U = np.exp(1j * phase)

    def reference(theta):
        tx, ty = np.deg2rad(theta)
        fx = np.sin(tx) / wavelength
        fy = np.sin(ty) / wavelength

        return np.exp(
            1j * 2 * np.pi * (fx * x_phys + fy * y_phys)
        )

    R1 = reference(theta1)
    R2 = reference(theta2)

    H1 = np.abs(U + R1)**2
    H2 = np.abs(U + R2)**2

    return H1.astype(np.float32), H2.astype(np.float32)


def main() -> None:
    """Generate a synthetic dataset of phase maps and their off-axis holograms.
    The data layout is::
        data_synth/
            train/   sample_00000/ ...
            val/     sample_00000/ ...
            test/    sample_00000/ ...
    Each ``sample_xxxxx`` folder contains:
        - ``data.npz``           : compressed arrays of gt_phase, hologram1, hologram2
        - ``gt_phase.png`` / ``hologram1.png`` / ``hologram2.png``
          for quick visual inspection (8-bit PNG).
    """
    out_dir = Path("data_synth")
    splits = [
        ("train", 5000, 0),
        ("val",   100, 100_000),
        ("test",  100, 200_000),
    ]

    for mode, _, _ in splits:
        (out_dir / mode).mkdir(parents=True, exist_ok=True)

    print("Generating synthetic dataset with FIXED reference beam angles...")

    for mode, count, seed_offset in splits:
        for idx in tqdm(range(count), desc=mode.capitalize()):
            # Fresh seeded RNG per sample → fully reproducible
            sample_rng = np.random.default_rng(seed_offset + idx)
            
            # Sinh phase map với Zernike
            gt_phase = generate_zernike_phase_map(
                shape=(256, 256), 
                max_phase=10 * np.pi, 
                min_modes=2,
                max_modes=8,
                rng=sample_rng
            )
            
            # Sinh Hologram 1 & 2
            H1, H2 = simulate_offaxis_holograms(gt_phase)
            
            # Tạo thư mục mẫu
            sample_dir = out_dir / mode / f"sample_{idx:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            # Save raw arrays – npz keeps them together and compresses them.
            np.savez_compressed(sample_dir / "data.npz",
                                gt_phase=gt_phase,
                                hologram1=H1,
                                hologram2=H2)
            
            # Also save PNGs for quick visual checks.
            for arr, name in [(gt_phase, "gt_phase"), (H1, "hologram1"), (H2, "hologram2")]:
                max_val = arr.max()
                if max_val > 0:
                    img = (arr / max_val * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(arr, dtype=np.uint8)
                    
                PIL.Image.fromarray(img).save(sample_dir / f"{name}.png")

    print(f"✅ Generated synthetic dataset at '{out_dir}/'")

if __name__ == "__main__":
    main()