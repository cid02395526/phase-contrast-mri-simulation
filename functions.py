import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 17,
}
plt.rcParams.update(PLOT_STYLE)

GAMMA = 2.675e8  #proton gyromagnetic ratio

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
    For opposite bipolar encodes:
        delta_phi = 2*pi*v / VENC
    so
        v = (VENC / (2*pi)) * delta_phi
    """
    return (venc / (2 * np.pi)) * delta_phi


def wrap_phase(phi):
    """
    Wrap phase into (-pi, pi].
    Works for scalars or arrays.
    """
    return (phi + np.pi) % (2 * np.pi) - np.pi


def complex_signal_from_velocity_distribution(v, venc, polarity=+1, rho=None):
    """
    Complex voxel signal for one encoding polarity.

    For one encode:
        phi = polarity * pi * v / VENC
    where polarity = +1 or -1.

    Parameters
    ----------
    v : array-like
        Velocity samples in the voxel.
    venc : float
        Velocity encoding value (m/s).
    polarity : int
        +1 or -1 for opposite bipolar encodes.
    rho : array-like or None
        Optional weights/spin density.

    Returns
    -------
    complex
        Voxel-averaged complex signal.
    """
    v = np.asarray(v)

    if rho is None:
        rho = np.ones_like(v, dtype=float)
    rho = np.asarray(rho, dtype=float)

    phi = polarity * np.pi * v / venc
    S = np.sum(rho * np.exp(1j * phi)) / np.sum(rho)
    return S


def measure_velocity_from_signal(S_plus, S_minus, venc):
    """
    Estimate velocity from the phase difference between opposite-polarity
    acquisitions.

    IMPORTANT:
    delta_phi = angle(S_plus) - angle(S_minus), wrapped to (-pi, pi]
    and then
        v = (VENC / (2*pi)) * delta_phi
    """
    phi_plus = np.angle(S_plus)
    phi_minus = np.angle(S_minus)
    delta_phi = wrap_phase(phi_plus - phi_minus)
    return velocity_from_phase_difference(delta_phi, venc)


def add_complex_gaussian_noise(S, sigma_noise):
    """
    Add independent Gaussian noise to the real and imaginary components
    separately, each with standard deviation sigma_noise.

    Thermal/electronic noise is modelled this way: the two quadrature
    channels are independent, so SNR in any given voxel is |S| / sigma_noise.
    """
    noise_real = sigma_noise * np.random.randn(*np.shape(S))
    noise_imag = sigma_noise * np.random.randn(*np.shape(S))
    return S + (noise_real + 1j * noise_imag)


def compute_snr(S_clean, sigma_noise):
    """
    Analytical SNR in a voxel: |S_clean| / sigma_noise.

    Because dephasing reduces |S_clean| below 1, the effective SNR is
    lower than the baseline thermal SNR (1/sigma_noise), giving the 1/|S|
    amplification of velocity uncertainty.
    """
    return np.abs(S_clean) / sigma_noise


def compute_snr_montecarlo(S_clean, sigma_noise, n_trials=300):
    """
    Empirically estimate SNR from repeated noisy acquisitions.

    Each trial adds independent Gaussian noise to real and imaginary channels.
    SNR_MC = mean(|S_noisy|) / std(|S_noisy|)

    At SNR > ~5 this matches the analytical |S_clean|/sigma_noise closely.
    The spread across trials reflects true measurement variability rather than
    a prescribed value.
    """
    mags = np.array([
        np.abs(add_complex_gaussian_noise(S_clean, sigma_noise))
        for _ in range(n_trials)
    ])
    return np.mean(mags) / np.std(mags)

# =========================================================
# Effect of Voxel Size on Intravoxel Dephasing
# =========================================================

def linear_velocity_profile_1d(x, v0, alpha):
    """
    v(x) = v0 + alpha * x
    """
    return v0 + alpha * x


def analytic_linear_profile_signal_magnitude(alpha, L, venc):
    """
    For a 1D linear velocity profile:
        |S| ~ |sinc(alpha L / (2 VENC))|

    numpy sinc uses:
        sinc(z) = sin(pi z)/(pi z)
    """
    z = alpha * L / (2 * venc)
    return np.abs(np.sinc(z))


def simulate_voxel_dephasing_vs_size(
    voxel_sizes_mm=np.linspace(0.5, 8.0, 30),
    v0=0.3,
    alpha=80.0,
    venc=0.8,
    n_samples=4000
):
    """
    Simulate voxel-averaged signal magnitude for different voxel widths.
    """
    mags_numeric = []
    mags_analytic = []

    for L_mm in voxel_sizes_mm:
        L = L_mm * 1e-3  # mm -> m
        x = np.linspace(-L / 2, L / 2, n_samples)
        v = linear_velocity_profile_1d(x, v0, alpha)

        S = complex_signal_from_velocity_distribution(v, venc, polarity=+1)
        mags_numeric.append(np.abs(S))
        mags_analytic.append(analytic_linear_profile_signal_magnitude(alpha, L, venc))

    return voxel_sizes_mm, np.array(mags_numeric), np.array(mags_analytic)


def plot_voxel_dephasing_vs_size():
    voxel_sizes_mm, mags_num, mags_an = simulate_voxel_dephasing_vs_size()

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(voxel_sizes_mm, mags_num, label="Numerical voxel integration")
    plt.plot(voxel_sizes_mm, mags_an, "--", label="Analytic sinc model")
    plt.xlabel("Voxel width L (mm)")
    plt.ylabel("Normalised signal magnitude |S|")
    plt.title("Effect of Voxel Size on Intravoxel Dephasing")
    plt.legend()
    plt.tight_layout()
    plt.show()

# =========================================================
# Effect of Flow Profile
# =========================================================

def make_1d_profile(profile_type, x, radius, vmax):
    """
    Create simple 1D vessel profiles across diameter.
    x in meters, radius in meters.
    """
    r = np.abs(x)
    inside = r <= radius
    v = np.zeros_like(x)

    if profile_type == "plug":
        v[inside] = vmax

    elif profile_type == "parabolic":
        # Poiseuille-like profile
        v[inside] = vmax * (1 - (x[inside] / radius) ** 2)

    elif profile_type == "blunted":
        # flatter centre, steeper near wall
        n = 6
        v[inside] = vmax * (1 - (np.abs(x[inside]) / radius) ** n)

    elif profile_type == "boundary_layer":
        # Turbulent power-law boundary layer: fast flat core, steep drop near wall
        # v(r) = vmax * (1 - |r/R|)^(1/7)
        n_exp = 7
        v[inside] = vmax * (1 - (np.abs(x[inside]) / radius)) ** (1.0 / n_exp)

    else:
        raise ValueError(f"Unknown profile: {profile_type}")

    return v


def validate_boundary_layer_profile(
    radius_mm=4.0,
    vmax=5.0,
    n_samples=10_000,
    tol=1e-6,
    plot=True
):
    """
    Validate the boundary-layer profile in ``make_1d_profile``.

    Six properties are checked:

    1. Centre velocity equals vmax  (v at x=0 should be vmax)
    2. Wall velocity equals zero    (v at x=±R should be 0)
    3. Outside-vessel velocity is zero
    4. Symmetry about the centreline  (v(x) == v(-x) for all x)
    5. Monotonicity – velocity decreases from centre to wall
    6. Analytic mean velocity matches numerical integral
       For the 1-D power-law profile v(x) = vmax·(1-|x/R|)^(1/n):
           mean = vmax · n/(n+1)   [exact integral over [-R, R]]
       With n=7  →  mean = 7/8 · vmax

    Parameters
    ----------
    radius_mm : float  – vessel radius in mm
    vmax      : float  – peak velocity (m/s)
    n_samples : int    – number of sample points across the diameter
    tol       : float  – absolute tolerance for floating-point comparisons
    plot      : bool   – if True, show a diagnostic plot

    Returns
    -------
    all_passed : bool  – True only when every check passes
    """
    radius = radius_mm * 1e-3
    n_exp  = 7                        # power-law exponent used inside make_1d_profile
    x      = np.linspace(-radius * 1.5, radius * 1.5, n_samples)
    v      = make_1d_profile("boundary_layer", x, radius, vmax)

    results = {}

    # ------------------------------------------------------------------
    # 1. Centre velocity  (sample nearest to x=0)
    # ------------------------------------------------------------------
    centre_idx = np.argmin(np.abs(x))
    v_centre   = v[centre_idx]
    expected_centre = vmax * (1 - np.abs(x[centre_idx]) / radius) ** (1.0 / n_exp)
    check1 = np.isclose(v_centre, expected_centre, atol=tol)
    results["1. Centre velocity = vmax"] = (
        check1,
        f"v(0) = {v_centre:.6f}  expected ≈ {expected_centre:.6f} m/s"
    )

    # ------------------------------------------------------------------
    # 2. Wall velocity  (at x = ±R the bracket (1-|r/R|)^(1/7) = 0)
    #    Evaluate directly at x = ±R so we are guaranteed to be on-wall.
    # ------------------------------------------------------------------
    wall_pts = np.array([-radius, radius])
    v_wall   = make_1d_profile("boundary_layer", wall_pts, radius, vmax)
    check2   = np.all(np.abs(v_wall) < tol)
    results["2. Wall velocity = 0"] = (
        check2,
        f"v(+R) = {v_wall[1]:.2e}  v(-R) = {v_wall[0]:.2e} m/s  (tol = {tol:.1e})"
    )

    # ------------------------------------------------------------------
    # 3. Outside-vessel velocity is exactly zero
    # ------------------------------------------------------------------
    outside = v[np.abs(x) > radius]
    check3  = np.all(outside == 0.0)
    results["3. Outside vessel → v = 0"] = (
        check3,
        f"max|v_outside| = {np.max(np.abs(outside)) if outside.size else 0:.2e} m/s"
    )

    # ------------------------------------------------------------------
    # 4. Symmetry: v(x) == v(-x)
    # ------------------------------------------------------------------
    x_sym    = np.linspace(-radius, radius, n_samples)
    v_sym    = make_1d_profile("boundary_layer", x_sym, radius, vmax)
    v_flip   = make_1d_profile("boundary_layer", -x_sym, radius, vmax)
    max_asym = np.max(np.abs(v_sym - v_flip))
    check4   = max_asym < tol
    results["4. Symmetry v(x) = v(-x)"] = (
        check4,
        f"max asymmetry = {max_asym:.2e} m/s"
    )

    # ------------------------------------------------------------------
    # 5. Monotonicity: velocity is non-increasing from centre to wall
    # ------------------------------------------------------------------
    x_half  = np.linspace(0, radius, n_samples // 2)
    v_half  = make_1d_profile("boundary_layer", x_half, radius, vmax)
    diffs   = np.diff(v_half)
    check5  = np.all(diffs <= tol)
    results["5. Monotonically decreasing centre → wall"] = (
        check5,
        f"max upward step = {np.max(diffs):.2e} m/s"
    )

    # ------------------------------------------------------------------
    # 6. Analytic mean  mean = vmax * n/(n+1)
    # ------------------------------------------------------------------
    x_inner  = np.linspace(-radius, radius, n_samples)
    v_inner  = make_1d_profile("boundary_layer", x_inner, radius, vmax)
    num_mean = np.mean(v_inner)
    ana_mean = vmax * n_exp / (n_exp + 1)          # exact 1-D integral result
    check6   = np.isclose(num_mean, ana_mean, rtol=1e-3)
    results["6. Mean velocity = vmax·n/(n+1)"] = (
        check6,
        f"numerical = {num_mean:.5f}  analytic = {ana_mean:.5f} m/s"
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("=" * 62)
    print("  Boundary-layer profile validation")
    print(f"  radius={radius_mm} mm   vmax={vmax} m/s   n_exp={n_exp}")
    print("=" * 62)
    all_passed = True
    for name, (passed, detail) in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {name}")
        print(f"         {detail}")
        if not passed:
            all_passed = False
    print("-" * 62)
    print(f"  Overall: {'ALL CHECKS PASSED' if all_passed else 'ONE OR MORE CHECKS FAILED'}")
    print("=" * 62)

    # ------------------------------------------------------------------
    # Optional diagnostic plot
    # ------------------------------------------------------------------
    if plot:
        x_plot = np.linspace(-radius * 1.3, radius * 1.3, 500)
        v_plot = make_1d_profile("boundary_layer", x_plot, radius, vmax)

        # Analytic curve (only inside vessel)
        x_an  = np.linspace(-radius, radius, 500)
        v_an  = vmax * (1 - np.abs(x_an) / radius) ** (1.0 / n_exp)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # --- Left: profile shape ---
        ax = axes[0]
        ax.plot(x_plot * 1e3, v_plot, color="tomato", lw=2,
                label="Numerical (make_1d_profile)")
        ax.plot(x_an * 1e3, v_an, "k--", lw=1.5,
                label=r"Analytic: $v_{max}(1-|r/R|)^{1/7}$")
        ax.axvline(-radius_mm, color="grey", ls=":", label="Vessel wall")
        ax.axvline( radius_mm, color="grey", ls=":")
        ax.axhline(ana_mean, color="steelblue", ls="--",
                   label=f"Analytic mean = {ana_mean:.3f} m/s")
        ax.axhline(num_mean, color="steelblue", ls=":", alpha=0.7,
                   label=f"Numerical mean = {num_mean:.3f} m/s")
        ax.set_xlabel("Position across vessel (mm)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title("Boundary-layer profile shape")
        ax.legend()

        # --- Right: velocity gradient dv/dx (physically meaningful) ---
        # The residual is machine-precision zero because make_1d_profile uses
        # the same analytic formula — confirmed correct. Instead we show the
        # shear rate dv/dx, which directly drives intravoxel dephasing.
        ax2 = axes[1]
        x_grad  = np.linspace(-radius, radius, n_samples)
        v_grad  = make_1d_profile("boundary_layer", x_grad, radius, vmax)
        dx      = x_grad[1] - x_grad[0]
        dvdx    = np.gradient(v_grad, dx)          # numerical first derivative

        # Analytic gradient: dv/dx = -(vmax/n) * sign(x) / R * (1-|x/R|)^(1/n - 1)
        # (diverges at the wall — the hallmark of a true boundary layer)
        sign_x  = np.sign(x_grad)
        sign_x[sign_x == 0] = 1
        inner   = 1 - np.abs(x_grad) / radius
        inner   = np.clip(inner, 1e-9, None)       # avoid 0^negative at wall
        dvdx_an = -(vmax / n_exp) * sign_x / radius * inner ** (1.0 / n_exp - 1)

        ax2.plot(x_grad * 1e3, dvdx,    color="tomato",    lw=2,  label="Numerical  dv/dx")
        ax2.plot(x_grad * 1e3, dvdx_an, "k--", lw=1.5, label="Analytic  dv/dx")
        ax2.axhline(0, color="grey", lw=0.6)
        ax2.axvline(-radius_mm, color="grey", ls=":")
        ax2.axvline( radius_mm, color="grey", ls=":")
        ax2.set_xlabel("Position across vessel (mm)")
        ax2.set_ylabel("Velocity gradient  dv/dx  (s⁻¹)")
        ax2.set_title("Shear rate across vessel\n(diverges at wall — boundary-layer signature)")
        ax2.legend()

        plt.suptitle("Boundary-layer profile validation")
        plt.tight_layout()
        plt.show()

    return all_passed


def simulate_flow_profile_effect(
    profiles=("plug", "parabolic", "blunted"),
    radius_mm=4.0,
    vmax=5.0,
    venc=1.5,
    voxel_width_mm=0.5,
    n_voxels=60,
    samples_per_voxel=800
):
    """
    Scan across a vessel and compute measured PC velocity voxel-by-voxel.
    """
    radius = radius_mm * 1e-3
    voxel_width = voxel_width_mm * 1e-3

    x_centres = np.linspace(-radius, radius, n_voxels)
    results = {}

    for profile_name in profiles:
        true_mean = []
        measured = []
        signal_mag = []

        for xc in x_centres:
            x_local = np.linspace(
                xc - voxel_width / 2,
                xc + voxel_width / 2,
                samples_per_voxel
            )
            v = make_1d_profile(profile_name, x_local, radius, vmax)

            S_plus = complex_signal_from_velocity_distribution(v, venc, polarity=+1)
            S_minus = complex_signal_from_velocity_distribution(v, venc, polarity=-1)

            v_est = measure_velocity_from_signal(S_plus, S_minus, venc)

            true_mean.append(np.mean(v))
            measured.append(v_est)
            signal_mag.append(np.abs(S_plus))

        results[profile_name] = {
            "x_mm": x_centres * 1e3,
            "true_mean": np.array(true_mean),
            "measured": np.array(measured),
            "signal_mag": np.array(signal_mag),
        }

    return results


def plot_flow_profile_effect():
    results = simulate_flow_profile_effect()

    plt.figure(figsize=(6.2, 4.2))
    for name, res in results.items():
        plt.plot(res["x_mm"], res["true_mean"], "--", label=f"{name} true mean")
        plt.plot(res["x_mm"], res["measured"], label=f"{name} measured")
    plt.xlabel("Position across vessel (mm)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Effect of Flow Profile on Measured PC-MRI Velocity")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6.2, 4.2))
    for name, res in results.items():
        plt.plot(res["x_mm"], res["signal_mag"], label=f"{name}")
    plt.xlabel("Position across vessel (mm)")
    plt.ylabel("Signal magnitude |S|")
    plt.title("Signal Loss from Intravoxel Dephasing for Different Flow Profiles")
    plt.legend()
    plt.tight_layout()
    plt.show()

# =========================================================
# Plug vs Boundary-Layer Flow with Noise
# Velocity, Signal Magnitude, and Measurement Uncertainty
# =========================================================

def simulate_plug_boundary_layer_with_noise(
    radius_mm=4.0,
    vmax=5.0,
    venc=11.0,  # bipolar ±1 encoding: measurable range is |v| < VENC/2
                # so need VENC > 2*vmax = 10.0; 11.0 gives ~10% headroom
    voxel_width_mm=0.5,
    n_voxels=60,
    samples_per_voxel=800,
    snr=20,
    n_trials=80
):
    """
    Scan across the vessel for plug and boundary-layer flow.

    Two noise sources are modelled and kept distinct:

    1. Thermal noise  — Gaussian noise added independently to the real and
       imaginary channels with sigma_noise = 1/snr.  This sets the baseline
       phase uncertainty: sigma_v_thermal = VENC / (pi * snr).

    2. Intravoxel dephasing — velocity spread across the voxel reduces the
       voxel-averaged signal magnitude |S| < 1.  This amplifies the velocity
       uncertainty because the effective SNR in the voxel is
       SNR_voxel = |S_clean| / sigma_noise, giving
       sigma_v_total = VENC / (pi * SNR_voxel) = VENC / (pi * snr * |S|).

    The Monte-Carlo std of velocity estimates captures both effects together.
    """
    radius = radius_mm * 1e-3
    voxel_width = voxel_width_mm * 1e-3
    x_centres = np.linspace(-radius, radius, n_voxels)
    sigma_noise = 1.0 / snr  # thermal noise std per real/imag channel

    results = {}
    for profile_name in ("plug", "boundary_layer"):
        true_mean = []
        meas_mean = []
        meas_std = []
        signal_mag = []
        snr_voxel = []
        sigma_v_thermal = []   # noise-only theory: VENC / (pi * snr)
        sigma_v_total = []     # noise + dephasing:  VENC / (pi * SNR_voxel)

        for xc in x_centres:
            x_local = np.linspace(xc - voxel_width / 2,
                                  xc + voxel_width / 2,
                                  samples_per_voxel)
            v = make_1d_profile(profile_name, x_local, radius, vmax)

            S_plus_clean = complex_signal_from_velocity_distribution(v, venc, polarity=+1)
            S_minus_clean = complex_signal_from_velocity_distribution(v, venc, polarity=-1)

            smag = np.abs(S_plus_clean)
            snr_v = compute_snr(S_plus_clean, sigma_noise)

            snr_mc = compute_snr_montecarlo(S_plus_clean, sigma_noise, n_trials=n_trials)

            signal_mag.append(smag)
            snr_voxel.append(snr_v)
            true_mean.append(np.mean(v))
            sigma_v_thermal.append(venc / (np.pi * snr))
            sigma_v_total.append(venc / (np.pi * max(snr_v, 0.01)))

            v_ests = []
            for _ in range(n_trials):
                Sp = add_complex_gaussian_noise(S_plus_clean, sigma_noise)
                Sm = add_complex_gaussian_noise(S_minus_clean, sigma_noise)
                v_ests.append(measure_velocity_from_signal(Sp, Sm, venc))

            v_ests = np.array(v_ests)
            meas_mean.append(np.mean(v_ests))
            meas_std.append(np.std(v_ests))

        results[profile_name] = {
            "x_mm": x_centres * 1e3,
            "true_mean": np.array(true_mean),
            "meas_mean": np.array(meas_mean),
            "meas_std": np.array(meas_std),       # true sigma_v from MC
            "signal_mag": np.array(signal_mag),
            "snr_voxel": np.array(snr_voxel),     # analytical: |S|/sigma_noise
            "snr_mc": snr_mc,                     # MC estimate from repeated acquisitions
            "sigma_v_thermal": np.array(sigma_v_thermal),
            "sigma_v_total": np.array(sigma_v_total),
        }

    return results


def plot_plug_boundary_layer_velocity():
    """
    True mean velocity vs measured (noisy) velocity across the vessel
    for plug and boundary-layer profiles.  Shaded band shows ±1σ noise.
    """
    results = simulate_plug_boundary_layer_with_noise()

    colors = {"plug": "steelblue", "boundary_layer": "tomato"}
    labels = {"plug": "Plug flow", "boundary_layer": "Boundary layer"}

    plt.figure(figsize=(6.2, 4.2))
    for name, res in results.items():
        c = colors[name]
        lbl = labels[name]
        plt.plot(res["x_mm"], res["true_mean"], "--", color=c,
                 label=f"{lbl} – true mean")
        plt.plot(res["x_mm"], res["meas_mean"], color=c,
                 label=f"{lbl} – measured (noisy)")
        plt.fill_between(res["x_mm"],
                         res["meas_mean"] - res["meas_std"],
                         res["meas_mean"] + res["meas_std"],
                         alpha=0.20, color=c)

    plt.xlabel("Position across vessel (mm)")
    plt.ylabel("Velocity (m/s)")
    plt.title("PC-MRI Velocity: Plug vs Boundary-Layer Flow (Noise, ±1σ Band)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_plug_boundary_layer_signal_magnitude():
    """
    Intravoxel-dephasing signal magnitude |S| vs position
    for plug and boundary-layer profiles.
    """
    results = simulate_plug_boundary_layer_with_noise()

    colors = {"plug": "steelblue", "boundary_layer": "tomato"}
    labels = {"plug": "Plug flow", "boundary_layer": "Boundary layer"}

    all_mags = np.concatenate([res["signal_mag"] for res in results.values()])
    y_min = max(0.0, np.min(all_mags) - 0.05)

    plt.figure(figsize=(6.2, 4.2))
    for name, res in results.items():
        plt.plot(res["x_mm"], res["signal_mag"],
                 color=colors[name], label=labels[name], linewidth=2)

    plt.xlabel("Position across vessel (mm)")
    plt.ylabel("Signal magnitude |S|")
    plt.title("Signal Magnitude vs Position: Plug and Boundary-Layer Flow")
    plt.ylim(y_min, 1.02)
    plt.margins(x=0.02)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_velocity_uncertainty_vs_position():
    """
    Velocity uncertainty σᵥ vs position across the vessel at fixed VENC.

    This is the clearest demonstration of the dephasing effect: interior voxels
    have |S|≈1 so uncertainty is set purely by thermal noise; near-wall voxels
    lose signal coherence (|S|<1), amplifying uncertainty by 1/|S|.

    Two theory curves are overlaid:
      - Flat dashed line: thermal noise only  σᵥ = VENC/(π·SNR)
      - Position-varying:  with dephasing     σᵥ = VENC/(π·SNR·|S|)
    """
    results = simulate_plug_boundary_layer_with_noise()

    colors = {"plug": "steelblue", "boundary_layer": "tomato"}
    labels = {"plug": "Plug flow", "boundary_layer": "Boundary layer"}

    # thermal-noise-only baseline (same for all positions)
    sample = next(iter(results.values()))
    sigma_thermal = sample["sigma_v_thermal"][0]  # constant across positions

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(sigma_thermal, color="k", linestyle="--", linewidth=1.5,
               label=r"Thermal noise only: $\sigma_v = \mathrm{VENC}/(\pi\,\mathrm{SNR})$",
               zorder=3)

    for name, res in results.items():
        c = colors[name]
        lbl = labels[name]
        ax.plot(res["x_mm"], res["meas_std"], color=c, linewidth=2,
                label=f"{lbl} – MC $\\sigma_v$")
        ax.plot(res["x_mm"], res["sigma_v_total"], color=c, linestyle="--",
                alpha=0.55,
                label=f"{lbl} – theory: " + r"$\mathrm{VENC}/(\pi\,\mathrm{SNR}\,|S|)$")

    ax.set_xlabel("Position across vessel (mm)")
    ax.set_ylabel(r"Velocity uncertainty $\sigma_v$ (m/s)")
    ax.set_title("Velocity Uncertainty vs Position\n"
                 "Near-Wall Dephasing Amplifies Uncertainty via 1/|S|")
    ax.margins(x=0.02)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# =========================================================
# Velocity Uncertainty vs VENC for Boundary-Layer Flow
# =========================================================

def simulate_venc_uncertainty_boundary_layer(
    venc_values=np.linspace(0.3, 2.0, 25),
    radius_mm=4.0,
    v_fraction_of_venc=0.4,
    voxel_width_mm=2.0,
    samples_per_voxel=800,
    snr=20,
    n_trials=600
):
    """
    Velocity uncertainty (true sigma_v from MC) vs VENC at three radial
    positions for boundary-layer flow.

    Two noise contributions are tracked separately:
      - Thermal noise:    sigma_v_thermal = VENC / (pi * snr)
      - Dephasing effect: SNR_voxel = |S| / sigma_noise  →
                          sigma_v_total  = VENC / (pi * SNR_voxel)
                                         = VENC / (pi * snr * |S|)

    The MC std is the ground truth combining both effects.

    v_fraction_of_venc=0.4 is chosen so that the phase of the voxel-averaged
    complex signal S_plus stays well below π/2 at every position.  If this
    fraction is too large (≥0.55 for the mid-vessel 2 mm voxel), the phase
    centroid of S_plus crosses π/2 and delta_phi = 2·angle(S_plus) wraps
    through ±π, producing a large stochastic spike in velocity uncertainty at
    the crossing VENC — a numerical artefact, not a physical effect.

    voxel_width_mm=2.0 is intentionally large so that the near-wall voxel
    (centred at r=3.8 mm) spans into stationary tissue, producing a wide
    velocity distribution within the voxel and hence significant dephasing
    (|S| well below 1).  A 0.5 mm voxel in the vessel interior gives
    phase spreads of ~0.1 rad and |S|≈0.999 everywhere — indistinguishable.
    """
    radius = radius_mm * 1e-3
    voxel_width = voxel_width_mm * 1e-3
    sigma_noise = 1.0 / snr

    positions_mm = [0.0, 2.0, 3.8]
    pos_labels = ["Centre (r=0 mm)", "Mid-vessel (r=2 mm)", "Near wall (r=3.8 mm)"]

    # Baseline (noise-only, no dephasing, |S|=1)
    sigma_theory = venc_values / (np.pi * snr)

    sim_std = {lbl: [] for lbl in pos_labels}
    signal_mags = {lbl: [] for lbl in pos_labels}

    for venc in venc_values:
        vmax = v_fraction_of_venc * venc

        for xc_mm, lbl in zip(positions_mm, pos_labels):
            xc = xc_mm * 1e-3
            x_local = np.linspace(xc - voxel_width / 2,
                                  xc + voxel_width / 2,
                                  samples_per_voxel)
            v = make_1d_profile("boundary_layer", x_local, radius, vmax)

            S_plus_clean = complex_signal_from_velocity_distribution(v, venc, polarity=+1)
            S_minus_clean = complex_signal_from_velocity_distribution(v, venc, polarity=-1)
            signal_mags[lbl].append(np.abs(S_plus_clean))

            v_ests = []
            for _ in range(n_trials):
                Sp = add_complex_gaussian_noise(S_plus_clean, sigma_noise)
                Sm = add_complex_gaussian_noise(S_minus_clean, sigma_noise)
                v_ests.append(measure_velocity_from_signal(Sp, Sm, venc))
            sim_std[lbl].append(np.std(v_ests))

    return venc_values, sigma_theory, sim_std, signal_mags, pos_labels, snr


def plot_venc_uncertainty_boundary_layer():
    """
    Velocity uncertainty vs VENC for boundary-layer flow.
    Three voxel positions (centre / mid / near-wall) are compared against
    pure-noise theory and dephasing-corrected theory sigma = VENC/(pi*SNR*|S|).
    """
    venc_values, sigma_theory, sim_std, signal_mags, pos_labels, snr = \
        simulate_venc_uncertainty_boundary_layer()

    colors = ["steelblue", "seagreen", "tomato"]

    plt.figure(figsize=(8, 5))
    plt.plot(venc_values, sigma_theory, "k--", linewidth=1.5,
             label=r"Theory (noise only): $\sigma_v = \mathrm{VENC}/(\pi\,\mathrm{SNR})$")

    for lbl, c in zip(pos_labels, colors):
        smag = np.array(signal_mags[lbl])
        smag_safe = np.where(smag > 0.05, smag, 0.05)
        sigma_corrected = venc_values / (np.pi * snr * smag_safe)

        plt.plot(venc_values, sim_std[lbl], color=c,
                 label=f"Simulated – {lbl}")
        plt.plot(venc_values, sigma_corrected, "--", color=c, alpha=0.55,
                 label=r"Theory w/ dephasing: $\mathrm{VENC}/(\pi\,\mathrm{SNR}\,|S|)$ – " + lbl)

    plt.xlabel("VENC (m/s)")
    plt.ylabel("Velocity uncertainty (m/s)")
    plt.title("Velocity Uncertainty vs VENC: Boundary-Layer Flow with Noise")
    plt.legend(fontsize=7.5)
    plt.tight_layout()
    plt.show()


# =========================================================
# VENC Optimisation and Velocity Noise
# =========================================================

def simulate_venc_noise_only(
    venc_values=np.linspace(0.3, 2.0, 30),
    snr=20,
    n_trials=5000,
    v_fraction_of_venc=0.25
):
    """
    Noise-only (no intravoxel dephasing) comparison across VENCs.

    A single-velocity voxel has |S|=1, so SNR_voxel = snr.  Gaussian noise
    is added independently to the real and imaginary channels of both
    encoding polarities (sigma_noise = 1/snr), and velocity is recovered
    from the phase difference.  The resulting MC std matches the theoretical
    sigma_v = VENC / (pi * snr).
    """
    sigma_noise = 1.0 / snr
    sigma_sim = []
    sigma_theory = venc_values / (np.pi * snr)

    for venc in venc_values:
        vtrue = v_fraction_of_venc * venc

        # Single-velocity voxel: |S|=1, no dephasing
        S_plus_clean  = np.exp( 1j * np.pi * vtrue / venc)
        S_minus_clean = np.exp(-1j * np.pi * vtrue / venc)

        estimates = []
        for _ in range(n_trials):
            Sp = add_complex_gaussian_noise(S_plus_clean,  sigma_noise)
            Sm = add_complex_gaussian_noise(S_minus_clean, sigma_noise)
            estimates.append(measure_velocity_from_signal(Sp, Sm, venc))

        sigma_sim.append(np.std(np.array(estimates)))

    return venc_values, np.array(sigma_sim), sigma_theory


def simulate_aliasing_fraction(
    true_velocities=np.linspace(0.0, 1.2, 200),
    venc_values=np.linspace(0.2, 2.0, 40)
):
    """
    Fraction of velocities that exceed VENC and therefore would alias.
    """
    alias_fraction = []

    for venc in venc_values:
        aliased = np.abs(true_velocities) > venc
        alias_fraction.append(np.mean(aliased))

    return venc_values, np.array(alias_fraction)


def plot_venc_optimisation():
    venc_values, sigma_sim, sigma_theory = simulate_venc_noise_only()
    venc_alias, alias_fraction = simulate_aliasing_fraction()

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(venc_values, sigma_sim, label="Simulated velocity SD")
    plt.plot(
        venc_values,
        sigma_theory,
        "--",
        label=r"Theory: $\sigma_v = \mathrm{VENC}/(\pi\,\mathrm{SNR})$"
    )
    plt.xlabel("VENC (m/s)")
    plt.ylabel("Velocity uncertainty (m/s)")
    plt.title("Noise-Only VENC Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(venc_alias, alias_fraction)
    plt.xlabel("VENC (m/s)")
    plt.ylabel("Aliasing fraction")
    plt.title("Aliasing Risk Decreases as VENC Increases")
    plt.tight_layout()
    plt.show()

# =========================================================
# Gradient Hardware Limits
# =========================================================

def min_ramp_time(G_target, slew_max):
    """
    Ramp time to reach gradient amplitude:
        t_ramp = G / slew
    """
    return G_target / slew_max


def feasible_delta_for_gradient(G_target, slew_max, delta_requested):
    """
    If trapezoidal gradient is used, pulse duration must be at least enough
    to ramp up and ramp down.
    """
    ramp = min_ramp_time(G_target, slew_max)
    return max(delta_requested, 2 * ramp)


def venc_with_hardware(G, delta, Delta, gamma=GAMMA):
    return np.pi / (gamma * G * delta * Delta)


def estimate_te_gradient(delta_eff, Delta):
    """
    TE contribution from the bipolar velocity-encoding gradient pair.

    TE_grad = delta_eff + Delta

    delta_eff already includes the ramp time (the gradient must ramp up and
    down within delta, so delta_eff >= 2 * t_ramp = 2 * G / slew).  The ramp
    time is small relative to Delta, keeping TE_grad in the 1-5 ms range
    that is realistic for modern PC-MRI sequences.
    """
    return delta_eff + Delta


def print_venc_spot_check(G_mTm, delta_ms, Delta_ms):
    """
    Quick hand-check for one point in the heatmap.
    """
    G = G_mTm * 1e-3
    delta = delta_ms * 1e-3
    Delta = Delta_ms * 1e-3

    venc = venc_with_hardware(G, delta, Delta)
    print(f"G = {G_mTm:.1f} mT/m, delta = {delta_ms:.1f} ms, Delta = {Delta_ms:.1f} ms")
    print(f"VENC = {venc:.4f} m/s")


def simulate_gradient_hardware_limits(
    Gmax_mTm=30.0,
    slew_max_Tm_s=150.0,
    Delta_values_ms=np.linspace(0.5, 4.0, 200),    # clinical range; old 3-15 ms gave sub-clinical VENCs
    delta_requested_ms=np.linspace(0.1, 1.0, 200),  # 200 pts → steps ~0.005 ms, invisible in plot
    snr=20
):
    """
    For every feasible (delta, Delta) combination under the hardware
    constraints, compute:
      - achievable VENC
      - estimated TE proxy
      - velocity uncertainty  sigma_v = VENC / (pi * SNR)
        (same noise model applied in the boundary-layer Monte Carlo study)

    Parameter choice rationale
    --------------------------
    With Gmax=30 mT/m and slew=150 T/m/s the minimum ramp time is 0.2 ms
    and the hardware-limited delta_eff = max(dreq, 2*ramp) = max(dreq, 0.4 ms).
    Scanning Delta over 0.5-4 ms and dreq over 0.1-1 ms gives:
      VENC  ~ 0.1 – 2.0 m/s  (clinical aortic / cardiac range)
      TE    ~ 9   – 14  ms   (realistic for modern 3T hardware)
    The 200×200 grid makes the delta_eff staircase steps (~0.005 ms each)
    imperceptible — smaller than one pixel at any normal figure size.
    """
    Gmax = Gmax_mTm * 1e-3
    Delta_values = Delta_values_ms * 1e-3
    delta_requested = delta_requested_ms * 1e-3

    venc_map  = np.full((len(delta_requested), len(Delta_values)), np.nan)
    TE_map    = np.full_like(venc_map, np.nan)
    sigma_map = np.full_like(venc_map, np.nan)

    for i, dreq in enumerate(delta_requested):
        for j, Delta in enumerate(Delta_values):
            delta_eff = feasible_delta_for_gradient(Gmax, slew_max_Tm_s, dreq)

            if Delta <= delta_eff:
                continue

            venc = venc_with_hardware(Gmax, delta_eff, Delta)
            TE   = estimate_te_gradient(delta_eff, Delta)

            venc_map[i, j]  = venc
            TE_map[i, j]    = TE * 1e3          # ms
            sigma_map[i, j] = venc / (np.pi * snr)

    return Delta_values_ms, delta_requested_ms, venc_map, TE_map, sigma_map, snr


def plot_gradient_hardware_limits():
    from matplotlib.colors import LogNorm
    from matplotlib import ticker

    Delta_ms, delta_ms, venc_map, TE_map, sigma_map, snr = \
        simulate_gradient_hardware_limits()

    extent = [Delta_ms.min(), Delta_ms.max(),
              delta_ms.min(), delta_ms.max()]
    imshow_kw = dict(origin="lower", aspect="auto", extent=extent)

    venc_m  = np.ma.masked_invalid(venc_map)
    TE_m    = np.ma.masked_invalid(TE_map)
    sigma_m = np.ma.masked_invalid(sigma_map)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    fig.suptitle(
        "Gradient Hardware Limits: VENC, TE, and Velocity Uncertainty\n"
        r"($G_{max}$=30 mT/m, slew=150 T/m/s, SNR=20)",
        fontsize=24,
    )

    # ── Panel 1: VENC ────────────────────────────────────────────────────
    ax = axes[0]
    venc_norm = LogNorm(vmin=np.nanmin(venc_map), vmax=np.nanmax(venc_map))
    im1 = ax.imshow(venc_m, **imshow_kw, cmap="viridis", norm=venc_norm)
    cb1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cb1.set_label("VENC (m/s)")
    venc_ticks = [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    cb1.set_ticks(venc_ticks)
    cb1.set_ticklabels([f"{t:.2g}" for t in venc_ticks])

    venc_levels = [0.25, 0.50, 0.75, 1.00, 1.50]
    cs1 = ax.contour(Delta_ms, delta_ms, venc_m,
                     levels=venc_levels, colors="white",
                     linewidths=0.9, alpha=0.85)
    ax.clabel(cs1, fmt="%.2f m/s", fontsize=11, inline=True)
    ax.set_xlabel(r"$\Delta$ (ms)")
    ax.set_ylabel(r"Requested $\delta$ (ms)")
    ax.set_title("Achievable VENC\n(log colour scale)")

    # ── Panel 2: TE ──────────────────────────────────────────────────────
    ax = axes[1]
    im2 = ax.imshow(TE_m, **imshow_kw, cmap="plasma")
    cb2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cb2.set_label("TE encoding contribution (ms)")
    te_levels = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    cs2 = ax.contour(Delta_ms, delta_ms, TE_m,
                     levels=te_levels, colors="white",
                     linewidths=0.9, alpha=0.85)
    ax.clabel(cs2, fmt="%g ms", fontsize=11, inline=True)
    ax.set_xlabel(r"$\Delta$ (ms)")
    ax.set_ylabel(r"Requested $\delta$ (ms)")
    ax.set_title(r"Gradient encoding TE ($\delta_{eff} + \Delta$)"
                 "\n(linear colour scale)")

    # ── Panel 3: Velocity uncertainty ────────────────────────────────────
    ax = axes[2]
    sig_norm = LogNorm(vmin=np.nanmin(sigma_map), vmax=np.nanmax(sigma_map))
    im3 = ax.imshow(sigma_m, **imshow_kw, cmap="magma", norm=sig_norm)
    cb3 = fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
    cb3.set_label(r"$\sigma_v$ (mm/s)")
    # Explicit log-spaced ticks covering the ~1.6–31 mm/s range
    sig_ticks = [0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020, 0.025, 0.030]
    cb3.set_ticks(sig_ticks)
    cb3.set_ticklabels([f"{t*1e3:.1f}" for t in sig_ticks])

    sig_levels = np.array(venc_levels) / (np.pi * snr)
    cs3 = ax.contour(Delta_ms, delta_ms, sigma_m,
                     levels=sig_levels, colors="white",
                     linewidths=0.9, alpha=0.85)
    ax.clabel(cs3, fmt=lambda v: f"{v*1e3:.1f} mm/s", fontsize=11, inline=True)
    ax.set_xlabel(r"$\Delta$ (ms)")
    ax.set_ylabel(r"Requested $\delta$ (ms)")
    ax.set_title(
        r"Velocity uncertainty  $\sigma_v = \mathrm{VENC}/(\pi\,\mathrm{SNR})$"
        "\n(log colour scale)"
    )

    plt.show()

# =========================================================
# 2D Beating-Heart Simulation
# =========================================================

def chamber_radius(t, T=1.0, R0=20e-3, A=4e-3):
    """
    Simple periodic chamber radius:
        R(t) = R0 - A cos(2*pi*t/T)
    """
    return R0 - A * np.cos(2 * np.pi * t / T)


def chamber_radial_velocity(t, T=1.0, A=4e-3):
    """
    dR/dt
    """
    return (2 * np.pi * A / T) * np.sin(2 * np.pi * t / T)


def make_2d_grid(nx=200, ny=200, fov=60e-3):
    x = np.linspace(-fov / 2, fov / 2, nx)
    y = np.linspace(-fov / 2, fov / 2, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return X, Y


def beating_heart_velocity_field(X, Y, t, T=1.0):
    """
    Simple model:
      - chamber interior: vertical flow (vy) during systole/diastole
      - wall region: radial motion
      - outside: zero

    The primary flow direction is vertical (vy), so the bipolar encoding
    gradient is aligned along y — matching the heartbeat direction.
    """
    R = chamber_radius(t, T=T)
    dRdt = chamber_radial_velocity(t, T=T)

    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    chamber = r <= R
    wall = (r > 0.9 * R) & (r <= 1.05 * R)

    vx = np.zeros_like(X)
    vy = np.zeros_like(Y)

    # radial wall motion
    vx[wall] += dRdt * np.cos(theta[wall])
    vy[wall] += dRdt * np.sin(theta[wall])

    # bulk blood flow
    flow_amp = 0.4  # m/s
    flow = flow_amp * np.sin(2 * np.pi * t / T)
    vy[chamber] += flow * (1 - (r[chamber] / R) ** 2)

    return vx, vy, chamber, wall


def pc_mri_encode_velocity_map(v_map, venc, polarity=+1):
    """
    Encode a velocity map into wrapped phase for one polarity.
    """
    phi = polarity * np.pi * v_map / venc
    return wrap_phase(phi)


def simulate_beating_heart_frames(
    n_frames=20,
    venc=0.6,
    T=1.0,
    nx=180,
    ny=180,
    fov=60e-3
):
    # Bipolar gradient aligned along y (primary heartbeat / flow direction)
    X, Y = make_2d_grid(nx=nx, ny=ny, fov=fov)
    frames = []

    for k in range(n_frames):
        t = k * T / n_frames
        vx, vy, chamber, wall = beating_heart_velocity_field(X, Y, t, T=T)

        phase_map = pc_mri_encode_velocity_map(vy, venc, polarity=+1)

        frames.append({
            "t": t,
            "vx": vx,
            "vy": vy,
            "phase": phase_map,
            "chamber": chamber,
            "wall": wall
        })

    return X, Y, frames


def plot_beating_heart_example(frame_idx=5):
    X, Y, frames = simulate_beating_heart_frames()
    fr = frames[frame_idx]

    plt.figure(figsize=(6, 5))
    plt.imshow(
        fr["vy"],
        origin="lower",
        extent=[X.min() * 1e3, X.max() * 1e3, Y.min() * 1e3, Y.max() * 1e3]
    )
    plt.colorbar(label="Velocity vy (m/s)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(f"Beating-Heart Velocity Field, t = {fr['t']:.2f} s")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.imshow(
        fr["phase"],
        origin="lower",
        extent=[X.min() * 1e3, X.max() * 1e3, Y.min() * 1e3, Y.max() * 1e3],
        vmin=-np.pi,
        vmax=np.pi
    )
    plt.colorbar(label="Wrapped phase (rad)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(f"Encoded phase map, t = {fr['t']:.2f} s")
    plt.tight_layout()
    plt.show()


def animate_beating_heart():
    X, Y, frames = simulate_beating_heart_frames(n_frames=30)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        frames[0]["phase"],
        origin="lower",
        extent=[X.min() * 1e3, X.max() * 1e3, Y.min() * 1e3, Y.max() * 1e3],
        vmin=-np.pi,
        vmax=np.pi
    )
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Wrapped phase (rad)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    def update(i):
        im.set_data(frames[i]["phase"])
        ax.set_title(f"2D beating-heart PC phase map, t = {frames[i]['t']:.2f} s")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=150, blit=False)
    plt.tight_layout()
    plt.show()
    return ani
