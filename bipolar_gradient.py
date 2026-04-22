import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Parameters (arbitrary units)
# --------------------------------------------------
gamma = 1.0
G = 1.0
delta = 1.0
Delta = 2.0  # start(lobe 1) -> start(lobe 2); choose Delta > delta to show a gap
v = 0.25

t_end = Delta + delta
pad_left = 0.4
pad_right = 1.3
t = np.linspace(-pad_left, t_end + pad_right, 4000)

# --------------------------------------------------
# Waveform and phase models
# --------------------------------------------------
def gradient_waveform(t_vals):
    g_vals = np.zeros_like(t_vals)
    g_vals[(t_vals >= 0.0) & (t_vals < delta)] = G
    g_vals[(t_vals >= Delta) & (t_vals < Delta + delta)] = -G
    return g_vals


def stationary_phase(t_vals):
    """Phase for v=0: phi = gamma * integral G(t') dt'."""
    phi = np.zeros_like(t_vals)

    lobe_1 = (t_vals >= 0.0) & (t_vals < delta)
    gap = (t_vals >= delta) & (t_vals < Delta)
    lobe_2 = (t_vals >= Delta) & (t_vals < Delta + delta)

    phi[lobe_1] = gamma * G * t_vals[lobe_1]
    phi[gap] = gamma * G * delta
    phi[lobe_2] = gamma * G * (delta - (t_vals[lobe_2] - Delta))

    return phi


def first_moment_accumulation(t_vals):
    """
    Running first-moment term used for velocity phase with this sign convention.
    By construction: m1_final = G * Delta * delta (positive for v > 0).
    """
    m1 = np.zeros_like(t_vals)

    lobe_1 = (t_vals >= 0.0) & (t_vals < delta)
    gap = (t_vals >= delta) & (t_vals < Delta)
    lobe_2 = (t_vals >= Delta) & (t_vals < Delta + delta)
    after = t_vals >= (Delta + delta)

    m1[lobe_1] = -0.5 * G * t_vals[lobe_1] ** 2
    m1[gap] = -0.5 * G * delta**2
    m1[lobe_2] = -0.5 * G * delta**2 + 0.5 * G * (t_vals[lobe_2] ** 2 - Delta**2)
    m1[after] = G * Delta * delta

    return m1


g = gradient_waveform(t)
phi_stat = stationary_phase(t)
m1_running = first_moment_accumulation(t)
phi_move = phi_stat + gamma * v * m1_running

phi_stat_final = 0.0
phi_move_final = gamma * G * delta * Delta * v

# --------------------------------------------------
# Style
# --------------------------------------------------
plt.rcParams.update(
    {
        # Fonts
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        # Axes / spines
        "axes.linewidth": 1.0,
        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5.0,
        "ytick.major.size": 5.0,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        # Lines
        "lines.linewidth": 1.5,
    }
)

col_grad = "#2166ac"
col_stat = "#555555"
col_move = "#c0392b"

# --------------------------------------------------
# Figure
# --------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(7.2, 3.8),            # slightly taller for A4 article readability
    gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.08},
    constrained_layout=True,
)

# ============================================================
# Panel (a): gradient waveform
# ============================================================
x_wave = np.array(
    [
        -pad_left,
        0.0,
        0.0,
        delta,
        delta,
        Delta,
        Delta,
        Delta + delta,
        Delta + delta,
        t_end + pad_right,
    ]
)
y_wave = np.array([0.0, 0.0, G, G, 0.0, 0.0, -G, -G, 0.0, 0.0])

ax1.spines["bottom"].set_position("zero")
ax1.fill_between([0.0, delta], [G, G], 0, color=col_grad, alpha=0.20)
ax1.fill_between([Delta, Delta + delta], [-G, -G], 0, color=col_grad, alpha=0.20)
ax1.plot(x_wave, y_wave, color=col_grad, linewidth=1.2)

ax1.text(delta / 2.0, 0.55 * G, r"$+G$", ha="center", va="center", color=col_grad, fontsize=12)
ax1.text(
    Delta + delta / 2.0,
    -0.55 * G,
    r"$-G$",
    ha="center",
    va="center",
    color=col_grad,
    fontsize=12,
)

y_delta = 1.22 * G
ax1.annotate(
    "",
    xy=(delta, y_delta),
    xytext=(0.0, y_delta),
    arrowprops=dict(arrowstyle="<->", lw=0.8, color="black"),
)
ax1.text(delta / 2.0, y_delta + 0.07 * G, r"$\delta$", ha="center", va="bottom", fontsize=13)

y_Delta = 1.85 * G
ax1.annotate(
    "",
    xy=(Delta, y_Delta),
    xytext=(0.0, y_Delta),
    arrowprops=dict(arrowstyle="<->", lw=0.8, color="black"),
)
ax1.text(Delta / 2.0, y_Delta + 0.07 * G, r"$\Delta$", ha="center", va="bottom", fontsize=13)

ax1.set_xlim(-pad_left, t_end + pad_right)
ax1.set_ylim(-1.45 * G, 2.3 * G)
ax1.set_yticks([-G, 0.0, G])
ax1.set_yticklabels([r"$-G$", "0", r"$+G$"])
ax1.set_xticks([])
ax1.set_xlabel("")
ax1.set_ylabel(r"$G(t)$")
ax1.annotate(r"Time $t$", xy=(t_end + pad_right, 0), xytext=(0, -6),
             textcoords="offset points", ha="right", va="top", fontsize=13)
ax1.set_title("(a) Bipolar gradient waveform", loc="left", pad=3)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ============================================================
# Panel (b): phase accumulation
# ============================================================
ax2.spines["bottom"].set_position("zero")
ax2.plot(t, phi_stat, color=col_stat, linewidth=1.2, label=r"Stationary spin ($v=0$)")
ax2.plot(
    t,
    phi_move,
    color=col_move,
    linewidth=1.2,
    linestyle="--",
    label=r"Moving spin ($v>0$)",
)

x_label = t_end + 0.18
scale = max(phi_stat.max(), phi_move.max(), phi_move_final, 1e-6)
stat_label_y = 0.05 * scale
move_label_y = phi_move_final + 0.11 * scale

ax2.text(x_label, stat_label_y, r"$\phi = 0$", color=col_stat, ha="left", va="bottom", fontsize=12)
ax2.text(
    x_label,
    move_label_y,
    r"$\phi = \gamma G \delta \Delta v$",
    color=col_move,
    ha="left",
    va="center",
    fontsize=12,
)

y_min = -0.04 * scale
y_max = max(phi_stat.max(), phi_move.max(), move_label_y) + 0.08 * scale

ax2.set_xlim(-pad_left, t_end + pad_right)
ax2.set_ylim(y_min, y_max)
ax2.set_xticks([])
ax2.set_yticks([0.0])
ax2.set_yticklabels(["0"])
ax2.set_xlabel("")
ax2.annotate(r"Time $t$", xy=(t_end + pad_right, 0), xytext=(0, -6),
             textcoords="offset points", ha="right", va="top", fontsize=13)
ax2.set_ylabel(r"Phase $\phi(t)$")
ax2.set_title("(b) Accumulated phase", loc="left", pad=3)
ax2.legend(loc="upper left", frameon=False, fontsize=8)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# --------------------------------------------------
# Save
# --------------------------------------------------
fig.savefig("bipolar_gradient.pdf", bbox_inches="tight")
fig.savefig("bipolar_gradient.png", bbox_inches="tight", dpi=600)

print(f"phi_stat final target: {phi_stat_final:.4f}")
print(f"phi_move final target: {phi_move_final:.4f}")
print("Saved bipolar_gradient.pdf and bipolar_gradient.png")

if "agg" not in plt.get_backend().lower():
    plt.show()
