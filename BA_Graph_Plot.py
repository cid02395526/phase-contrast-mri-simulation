# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 00:59:05 2026

@author: jamie
"""

import numpy as np
import matplotlib.pyplot as plt

GAMMA = 2.675e8  # proton gyromagnetic ratio


# =========================================================
# Core PC-MRI functions
# =========================================================

def phase_from_velocity(v, G, delta, Delta, gamma=GAMMA):
    """
    Motion-encoded phase for constant velocity:
        phi = gamma * G * delta * Delta * v
    """
    return gamma * G * delta * Delta * v


def venc_from_gradients(G, delta, Delta, gamma=GAMMA):
    """
    VENC = pi / (gamma * G * delta * Delta)
    """
    return np.pi / (gamma * G * delta * Delta)


def velocity_from_phase_difference(delta_phi, venc):
    """
    With the encoding convention used here:
        phi_(+) = + pi * v / VENC
        phi_(-) = - pi * v / VENC

    so:
        delta_phi = 2*pi*v / VENC

    hence:
        v = (VENC / (2*pi)) * delta_phi
    """
    return (venc / (2 * np.pi)) * delta_phi


def wrap_phase(phi):
    """
    Wrap phase into (-pi, pi].
    """
    return (phi + np.pi) % (2 * np.pi) - np.pi


def complex_signal_from_velocity_distribution(v, venc, polarity=+1, rho=None):
    """
    Complex voxel signal for one encoding polarity.

    For one encode:
        phi = polarity * pi * v / VENC
    """
    v = np.asarray(v, dtype=float)

    if rho is None:
        rho = np.ones_like(v, dtype=float)
    rho = np.asarray(rho, dtype=float)

    phi = polarity * np.pi * v / venc
    S = np.sum(rho * np.exp(1j * phi)) / np.sum(rho)
    return S


def measure_velocity_from_signal(S_plus, S_minus, venc):
    """
    Estimate velocity from opposite-polarity acquisitions.
    """
    phi_plus = np.angle(S_plus)
    phi_minus = np.angle(S_minus)
    delta_phi = wrap_phase(phi_plus - phi_minus)
    return velocity_from_phase_difference(delta_phi, venc)


def add_complex_gaussian_noise(S, sigma_noise):
    """
    Add independent Gaussian noise to real and imaginary parts.
    """
    noise_real = sigma_noise * np.random.randn(*np.shape(S))
    noise_imag = sigma_noise * np.random.randn(*np.shape(S))
    return S + (noise_real + 1j * noise_imag)


def compute_snr(S_clean, sigma_noise):
    """
    Analytical voxel SNR = |S_clean| / sigma_noise.
    """
    return np.abs(S_clean) / sigma_noise


# =========================================================
# Flow profiles
# =========================================================

def make_1d_profile(profile_type, x, radius, vmax):
    """
    Create 1D vessel velocity profiles across the diameter.
    x in meters, radius in meters.
    """
    r = np.abs(x)
    inside = r <= radius
    v = np.zeros_like(x, dtype=float)

    if profile_type == "plug":
        v[inside] = vmax

    elif profile_type == "parabolic":
        v[inside] = vmax * (1 - (x[inside] / radius) ** 2)

    elif profile_type == "blunted":
        n = 6
        v[inside] = vmax * (1 - (np.abs(x[inside]) / radius) ** n)

    elif profile_type == "boundary_layer":
        # Turbulent-like power-law boundary-layer profile
        # v(r) = vmax * (1 - |r/R|)^(1/7)
        n_exp = 7
        v[inside] = vmax * (1 - (np.abs(x[inside]) / radius)) ** (1.0 / n_exp)

    else:
        raise ValueError(f"Unknown profile: {profile_type}")

    return v


# =========================================================
# Noisy flow-profile simulation
# =========================================================

def simulate_flow_profile_with_noise(
    profile="boundary_layer",
    radius_mm=4.0,
    vmax=5.0,
    venc=11.0,
    voxel_width_mm=0.5,
    n_voxels=60,
    samples_per_voxel=800,
    snr=20,
    n_trials=80,
    seed=12345
):
    """
    Simulate noisy PC-MRI measurements across a vessel.

    Returns for each voxel:
      - true_mean: true voxel-mean velocity
      - meas_mean: mean reconstructed velocity over repeated noisy trials
      - meas_std:  std of reconstructed velocity over repeated noisy trials
      - signal_mag: clean voxel signal magnitude
    """
    np.random.seed(seed)

    radius = radius_mm * 1e-3
    voxel_width = voxel_width_mm * 1e-3
    sigma_noise = 1.0 / snr

    x_centres = np.linspace(-radius, radius, n_voxels)

    true_mean = []
    meas_mean = []
    meas_std = []
    signal_mag = []
    snr_voxel = []

    for xc in x_centres:
        x_local = np.linspace(
            xc - voxel_width / 2,
            xc + voxel_width / 2,
            samples_per_voxel
        )

        v = make_1d_profile(profile, x_local, radius, vmax)

        S_plus_clean = complex_signal_from_velocity_distribution(v, venc, polarity=+1)
        S_minus_clean = complex_signal_from_velocity_distribution(v, venc, polarity=-1)

        signal_mag.append(np.abs(S_plus_clean))
        snr_voxel.append(compute_snr(S_plus_clean, sigma_noise))
        true_mean.append(np.mean(v))

        v_ests = []
        for _ in range(n_trials):
            Sp = add_complex_gaussian_noise(S_plus_clean, sigma_noise)
            Sm = add_complex_gaussian_noise(S_minus_clean, sigma_noise)
            v_ests.append(measure_velocity_from_signal(Sp, Sm, venc))

        v_ests = np.array(v_ests)
        meas_mean.append(np.mean(v_ests))
        meas_std.append(np.std(v_ests))

    return {
        "x_mm": x_centres * 1e3,
        "true_mean": np.array(true_mean),
        "meas_mean": np.array(meas_mean),
        "meas_std": np.array(meas_std),
        "signal_mag": np.array(signal_mag),
        "snr_voxel": np.array(snr_voxel),
    }


# =========================================================
# Optional diagnostic plot
# =========================================================

def plot_velocity_profile_with_noise(
    profile="boundary_layer",
    radius_mm=4.0,
    vmax=5.0,
    venc=11.0,
    voxel_width_mm=0.5,
    n_voxels=60,
    samples_per_voxel=800,
    snr=20,
    n_trials=80,
    seed=12345
):
    """
    Plot true mean and noisy measured mean across vessel.
    """
    res = simulate_flow_profile_with_noise(
        profile=profile,
        radius_mm=radius_mm,
        vmax=vmax,
        venc=venc,
        voxel_width_mm=voxel_width_mm,
        n_voxels=n_voxels,
        samples_per_voxel=samples_per_voxel,
        snr=snr,
        n_trials=n_trials,
        seed=seed
    )

    plt.figure(figsize=(8, 5))
    plt.plot(res["x_mm"], res["true_mean"], "--", label="True mean velocity")
    plt.plot(res["x_mm"], res["meas_mean"], label="Measured mean velocity")
    plt.fill_between(
        res["x_mm"],
        res["meas_mean"] - res["meas_std"],
        res["meas_mean"] + res["meas_std"],
        alpha=0.2,
        label=r"Measured $\pm 1\sigma$"
    )
    plt.xlabel("Position across vessel (mm)")
    plt.ylabel("Velocity (m/s)")
    plt.title(f"PC-MRI velocity across vessel: {profile} flow")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return res


# =========================================================
# Bland–Altman analysis
# =========================================================

def bland_altman_stats(true_vals, measured_vals):
    """
    Compute Bland–Altman quantities.
    """
    true_vals = np.asarray(true_vals, dtype=float)
    measured_vals = np.asarray(measured_vals, dtype=float)

    means = 0.5 * (true_vals + measured_vals)
    diffs = measured_vals - true_vals

    bias = np.mean(diffs)
    sd = np.std(diffs, ddof=1)

    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    return means, diffs, bias, loa_lower, loa_upper


def plot_bland_altman(
    profile="boundary_layer",
    radius_mm=4.0,
    vmax=5.0,
    venc=11.0,
    voxel_width_mm=0.5,
    n_voxels=60,
    samples_per_voxel=800,
    snr=20,
    n_trials=80,
    exclude_edge_fraction=0.95,
    seed=12345,
    save_path=None
):
    """
    Bland–Altman plot using:
      - true voxel-mean velocity
      - noisy reconstructed mean velocity (averaged over Monte Carlo trials)
    """
    res = simulate_flow_profile_with_noise(
        profile=profile,
        radius_mm=radius_mm,
        vmax=vmax,
        venc=venc,
        voxel_width_mm=voxel_width_mm,
        n_voxels=n_voxels,
        samples_per_voxel=samples_per_voxel,
        snr=snr,
        n_trials=n_trials,
        seed=seed
    )

    x_mm = res["x_mm"]
    true_vals = res["true_mean"]
    measured_vals = res["meas_mean"]

    # Exclude extreme outer edge voxels if desired
    mask = np.abs(x_mm) <= exclude_edge_fraction * radius_mm
    x_mm = x_mm[mask]
    true_vals = true_vals[mask]
    measured_vals = measured_vals[mask]

    means, diffs, bias, loa_lower, loa_upper = bland_altman_stats(true_vals, measured_vals)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.scatter(
        means,
        diffs,
        s=38,
        color="tab:blue",
        edgecolors="black",
        linewidths=0.3,
        alpha=0.9,
        label="Voxel pairs"
    )

    ax.axhline(
        bias,
        color="crimson",
        linestyle="-",
        linewidth=1.8,
        label=f"Bias = {bias:.2e} m/s"
    )
    ax.axhline(
        loa_upper,
        color="darkgreen",
        linestyle="--",
        linewidth=1.6,
        label=f"Upper LoA = {loa_upper:.2e} m/s"
    )
    ax.axhline(
        loa_lower,
        color="darkgreen",
        linestyle="--",
        linewidth=1.6,
        label=f"Lower LoA = {loa_lower:.2e} m/s"
    )

    pretty_name = profile.replace("_", "-")
    ax.set_title(f"Bland–Altman analysis of simulated {pretty_name} flow", fontsize=12, pad=10)
    ax.set_xlabel("Mean of true and reconstructed velocity (m/s)", fontsize=11)
    ax.set_ylabel("Difference (reconstructed - true) (m/s)", fontsize=11)

    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    legend = ax.legend(loc="upper right", frameon=True, fontsize=9)
    legend.get_frame().set_alpha(0.95)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    print(f"Profile = {profile}")
    print(f"Number of points = {len(means)}")
    print(f"Bias = {bias:.6e} m/s")
    print(f"95% limits of agreement = [{loa_lower:.6e}, {loa_upper:.6e}] m/s")

    return {
        "x_mm": x_mm,
        "true_vals": true_vals,
        "measured_vals": measured_vals,
        "means": means,
        "diffs": diffs,
        "bias": bias,
        "loa_lower": loa_lower,
        "loa_upper": loa_upper,
    }


# =========================================================
# Example runs
# =========================================================

if __name__ == "__main__":
    # Optional check of the profile itself
    plot_velocity_profile_with_noise(
        profile="boundary_layer",
        radius_mm=4.0,
        vmax=5.0,
        venc=11.0,
        voxel_width_mm=0.5,
        n_voxels=60,
        samples_per_voxel=800,
        snr=20,
        n_trials=80,
        seed=12345
    )

    # Bland–Altman plot
    ba = plot_bland_altman(
        profile="boundary_layer",
        radius_mm=4.0,
        vmax=5.0,
        venc=11.0,
        voxel_width_mm=0.5,
        n_voxels=60,
        samples_per_voxel=800,
        snr=20,
        n_trials=80,
        exclude_edge_fraction=0.95,
        seed=12345,
        save_path="bland_altman_boundary_layer.png"
    )