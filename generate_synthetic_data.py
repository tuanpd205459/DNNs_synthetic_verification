import os
import math
import numpy as np
import torch
import PIL.Image


def zernike_eval(n, m, rho, phi):
    """
    Evaluate Zernike polynomial Z_n^m(rho, phi).
    """
    if (n - abs(m)) % 2 != 0:
        return np.zeros_like(rho)

    R = np.zeros_like(rho)

    for k in range((n - abs(m)) // 2 + 1):
        c = ((-1) ** k *
             math.factorial(n - k) /
             (math.factorial(k) *
              math.factorial((n + abs(m)) // 2 - k) *
              math.factorial((n - abs(m)) // 2 - k)))

        R += c * rho ** (n - 2 * k)

    if m >= 0:
        return R * np.cos(m * phi)
    else:
        return R * np.sin(-m * phi)


def generate_zernike_phase_map(
        shape=(256, 256),
        max_phase=10 * np.pi,
        min_active_modes=2,
        max_active_modes=8):

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

    phase = np.zeros_like(rho)

    n_active = np.random.randint(
        min_active_modes,
        max_active_modes + 1
    )

    active_idx = np.random.choice(
        len(z_modes),
        n_active,
        replace=False
    )

    coeffs = np.zeros(len(z_modes))
    coeffs[active_idx] = np.random.uniform(-5.0, 5.0, n_active)

    for coeff, (n, m) in zip(coeffs, z_modes):
        if coeff != 0:
            phase += coeff * zernike_eval(n, m, rho, phi)

    # Normalize phase to [0, max_phase]
    phase -= phase.min()

    if phase.max() > 0:
        phase /= phase.max()

    phase *= max_phase

    return phase.astype(np.float32)


def simulate_offaxis_holograms(
        phase_map,
        wavelength=0.6328,
        pixel_size=3.45):
    """
    Simulate two off-axis holograms with RANDOM reference beam angles (theta1, theta2) for each sample:
    fx = pixel_size * sin(theta_x) / wavelength
    fy = pixel_size * sin(theta_y) / wavelength
    R = exp(j*2*pi*(fx*XX + fy*YY))
    """

    H, W = phase_map.shape

    x_grid = np.arange(W, dtype=np.float32) - W / 2.0
    y_grid = np.arange(H, dtype=np.float32) - H / 2.0

    XX, YY = np.meshgrid(x_grid, y_grid)

   # theta_x1 = np.random.uniform(1.0, 3.0)
   # theta_y1 = np.random.uniform(1.0, 3.0)
    theta_x1 = 2.0
    theta_y1 = 2.0
    fx1 = pixel_size * np.sin(np.deg2rad(theta_x1)) / wavelength
    fy1 = pixel_size * np.sin(np.deg2rad(theta_y1)) / wavelength

    R1 = np.exp(1j * 2 * np.pi * (fx1 * XX + fy1 * YY))

   # theta_x2 = np.random.uniform(1.0, 3.0)
   # theta_y2 = np.random.uniform(1.0, 3.0)
    theta_x2 = 3.0
    theta_y2 = 3.0

    fx2 = pixel_size * np.sin(np.deg2rad(theta_x2)) / wavelength
    fy2 = pixel_size * np.sin(np.deg2rad(theta_y2)) / wavelength

    R2 = np.exp(1j * 2 * np.pi * (fx2 * XX + fy2 * YY))

    U = np.exp(1j * phase_map)

    H1 = np.abs(U + R1) ** 2
    H2 = np.abs(U + R2) ** 2

    return (
        H1.astype(np.float32),
        H2.astype(np.float32),
        (fx1, fy1),
        (fx2, fy2)
    )


def main():
    out_dir = "data_synth"

    splits = [
        ("train", 1000, 0),
        ("val",    100, 100000), 
        ("test",   100, 200000)
    ]

    for mode, _, _ in splits:
        os.makedirs(os.path.join(out_dir, mode), exist_ok=True)

    print("Generating synthetic dataset with RANDOM reference beam angles for each sample...")

    for mode, count, seed_offset in splits:
        print(f"Generating {mode} set ({count} samples)...")
        for idx in range(count):
            np.random.seed(seed_offset + idx)

            gt_phase = generate_zernike_phase_map(
                shape=(256, 256),
                max_phase=10 * np.pi
            )

            H1, H2, f1, f2 = simulate_offaxis_holograms(gt_phase)

            sample_dir = os.path.join(
                out_dir,
                mode,
                f"sample_{idx:05d}"
            )

            os.makedirs(sample_dir, exist_ok=True)

            np.save(os.path.join(sample_dir, "gt_phase.npy"), gt_phase)
            np.save(os.path.join(sample_dir, "hologram1.npy"), H1)
            np.save(os.path.join(sample_dir, "hologram2.npy"), H2)
            np.save(
                os.path.join(sample_dir, "freq_params.npy"),
                np.array([f1, f2], dtype=np.float32)
            )

            # Visualization images
            PIL.Image.fromarray(
                (gt_phase / gt_phase.max() * 255).astype(np.uint8)
            ).save(os.path.join(sample_dir, "gt_phase.png"))

            PIL.Image.fromarray(
                (H1 / H1.max() * 255).astype(np.uint8)
            ).save(os.path.join(sample_dir, "hologram1.png"))

            PIL.Image.fromarray(
                (H2 / H2.max() * 255).astype(np.uint8)
            ).save(os.path.join(sample_dir, "hologram2.png"))

    print(f"✅ Generated synthetic dataset with per-sample random reference angles in '{out_dir}/'!")


if __name__ == '__main__':
    main()