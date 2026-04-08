import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --------------------------------------------------
# 3.1 Simulation Framework — core signal functions
# --------------------------------------------------

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


def add_complex_gaussian_noise(S, snr_mag):
    """
    Add complex Gaussian noise.
    If |S| ~ 1, snr_mag approximately sets signal-to-noise ratio in magnitude.
    """
    sigma = 1.0 / snr_mag
    noise = sigma / np.sqrt(2) * (
        np.random.randn(*np.shape(S)) + 1j * np.random.randn(*np.shape(S))
    )
    return S + noise

# --------------------------------------------------
# 3.2 Effect of Voxel Size on Intravoxel Dephasing
# --------------------------------------------------

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

    plt.figure(figsize=(7, 4.5))
    plt.plot(voxel_sizes_mm, mags_num, label="Numerical voxel integration")
    plt.plot(voxel_sizes_mm, mags_an, "--", label="Analytic sinc model")
    plt.xlabel("Voxel width L (mm)")
    plt.ylabel("Normalised signal magnitude |S|")
    plt.title("Effect of Voxel Size on Intravoxel Dephasing")
    plt.legend()
    plt.tick_params(direction="in")
    plt.tight_layout()
    plt.savefig("fig_voxel_dephasing.png", bbox_inches="tight", dpi=150)
    plt.show()

# --------------------------------------------------
# 3.3 Effect of Flow Profile
# --------------------------------------------------

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

    else:
        raise ValueError(f"Unknown profile: {profile_type}")

    return v


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

    plt.figure(figsize=(7, 4.5))
    for name, res in results.items():
        plt.plot(res["x_mm"], res["true_mean"], "--", label=f"{name} true mean")
        plt.plot(res["x_mm"], res["measured"], label=f"{name} measured")
    plt.xlabel("Position across vessel (mm)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Effect of Flow Profile on Measured PC-MRI Velocity")
    plt.legend()
    plt.tick_params(direction="in")
    plt.tight_layout()
    plt.savefig("fig_flow_profiles.png", bbox_inches="tight", dpi=150)
    plt.show()

    plt.figure(figsize=(7, 4.5))
    for name, res in results.items():
        plt.plot(res["x_mm"], res["signal_mag"], label=f"{name}")
    plt.xlabel("Position across vessel (mm)")
    plt.ylabel("Signal magnitude |S|")
    plt.title("Signal Loss from Intravoxel Dephasing for Different Flow Profiles")
    plt.legend()
    plt.tick_params(direction="in")
    plt.tight_layout()
    plt.savefig("fig_signal_loss_profiles.png", bbox_inches="tight", dpi=150)
    plt.show()

# --------------------------------------------------
# 3.4 VENC Optimisation and Velocity Noise
# --------------------------------------------------

def simulate_venc_noise_only(
    venc_values=np.linspace(0.3, 2.0, 30),
    snr=20,
    n_trials=5000,
    v_fraction_of_venc=0.25
):
    """
    Noise-only comparison:
    choose vtrue as a fixed fraction of VENC so the phase stays away from wrapping.
    """
    sigma_sim = []
    sigma_theory = venc_values / (np.pi * snr)

    for venc in venc_values:
        vtrue = v_fraction_of_venc * venc

        # phase difference between opposite encodes:
        # delta_phi = 2*pi*v / VENC
        delta_phi_true = 2 * np.pi * vtrue / venc

        estimates = []
        for _ in range(n_trials):
            delta_phi_noisy = delta_phi_true + np.random.normal(0, 1 / snr)
            delta_phi_wrapped = wrap_phase(delta_phi_noisy)

            vest = velocity_from_phase_difference(delta_phi_wrapped, venc)
            estimates.append(vest)

        estimates = np.array(estimates)
        sigma_sim.append(np.std(estimates))

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

    plt.figure(figsize=(7, 4.5))
    plt.plot(venc_values, sigma_sim, label="Simulated velocity SD")
    plt.plot(
        venc_values,
        sigma_theory,
        "--",
        label=r"Theory: $\sigma_v = \mathrm{VENC}/(\pi\,\mathrm{SNR})$"
    )
    plt.xlabel("VENC (m/s)")
    plt.ylabel("Velocity uncertainty (m/s)")
    plt.title("Noise-only VENC Comparison")
    plt.legend()
    plt.tick_params(direction="in")
    plt.tight_layout()
    plt.savefig("fig_venc_noise.png", bbox_inches="tight", dpi=150)
    plt.show()

    plt.figure(figsize=(7, 4.5))
    plt.plot(venc_alias, alias_fraction)
    plt.xlabel("VENC (m/s)")
    plt.ylabel("Aliasing fraction")
    plt.title("Aliasing Risk Decreases as VENC Increases")
    plt.tick_params(direction="in")
    plt.tight_layout()
    plt.savefig("fig_aliasing.png", bbox_inches="tight", dpi=150)
    plt.show()

# --------------------------------------------------
# 3.5 Gradient Hardware Limits
# --------------------------------------------------

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


def estimate_echo_time_proxy(delta, Delta, t_readout=5e-3, t_misc=3e-3):
    """
    Very rough TE proxy:
      TE ~ 2*delta + Delta + readout + misc overhead
    """
    return 2 * delta + Delta + t_readout + t_misc


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
    Delta_values_ms=np.linspace(3, 15, 50),
    delta_requested_ms=np.linspace(0.5, 5.0, 50)
):
    Gmax = Gmax_mTm * 1e-3
    Delta_values = Delta_values_ms * 1e-3
    delta_requested = delta_requested_ms * 1e-3

    venc_map = np.full((len(delta_requested), len(Delta_values)), np.nan)
    TE_map = np.full_like(venc_map, np.nan)

    for i, dreq in enumerate(delta_requested):
        for j, Delta in enumerate(Delta_values):
            delta_eff = feasible_delta_for_gradient(Gmax, slew_max_Tm_s, dreq)

            if Delta <= delta_eff:
                continue

            venc = venc_with_hardware(Gmax, delta_eff, Delta)
            TE = estimate_echo_time_proxy(delta_eff, Delta)

            venc_map[i, j] = venc
            TE_map[i, j] = TE * 1e3  # ms

    return Delta_values_ms, delta_requested_ms, venc_map, TE_map


def plot_gradient_hardware_limits():
    Delta_ms, delta_ms, venc_map, TE_map = simulate_gradient_hardware_limits()

    plt.figure(figsize=(7, 5))
    im = plt.imshow(
        venc_map,
        origin="lower",
        aspect="auto",
        extent=[Delta_ms.min(), Delta_ms.max(), delta_ms.min(), delta_ms.max()]
    )
    plt.xlabel(r"$\Delta$ (ms)")
    plt.ylabel(r"Requested $\delta$ (ms)")
    plt.title("Achievable VENC under Gradient Hardware Limits")
    plt.colorbar(im, label="VENC (m/s)")
    plt.tick_params(direction="in")
    plt.tight_layout()
    plt.savefig("fig_venc_hardware.png", bbox_inches="tight", dpi=150)
    plt.show()

    plt.figure(figsize=(7, 5))
    im = plt.imshow(
        TE_map,
        origin="lower",
        aspect="auto",
        extent=[Delta_ms.min(), Delta_ms.max(), delta_ms.min(), delta_ms.max()]
    )
    plt.xlabel(r"$\Delta$ (ms)")
    plt.ylabel(r"Requested $\delta$ (ms)")
    plt.title("Estimated TE Proxy under Hardware-Limited Encoding")
    plt.colorbar(im, label="TE (ms)")
    plt.tick_params(direction="in")
    plt.tight_layout()
    plt.savefig("fig_te_hardware.png", bbox_inches="tight", dpi=150)
    plt.show()

# --------------------------------------------------
# 3.6 2D Beating Heart Simulation
# --------------------------------------------------

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
      - chamber interior: vertical flow during systole/diastole
      - wall region: radial motion
      - outside: zero
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
    plt.title(f"Beating-heart velocity field, t = {fr['t']:.2f} s")
    plt.tick_params(direction="in")
    plt.tight_layout()
    plt.savefig("fig_heart_velocity.png", bbox_inches="tight", dpi=150)
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
    plt.tick_params(direction="in")
    plt.tight_layout()
    plt.savefig("fig_heart_phase.png", bbox_inches="tight", dpi=150)
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