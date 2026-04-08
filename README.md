# Simulating Phase Contrast MRI for Quantifying Blood Flow in the Cardiovascular System

Supplementary code for the report produced for the **Physics of Medical Imaging and Radiotherapy** course at **Imperial College London**, April 2026.

**Authors:** Mingyu Liang, Zak Nabi, Jamie Ward  
**Supervisor:** Dr. Andrew Scott

---

## Overview

This repository contains the Python simulation framework developed to investigate phase contrast MRI (PC-MRI) under realistic cardiovascular conditions. The code models PC-MRI signal formation from blood flowing through vessels and cardiac chambers, allowing the effects of key acquisition parameters to be studied under controlled conditions where the ground-truth velocity field is known exactly.

The simulations cover:

- Bipolar gradient encoding and phase accumulation
- Intravoxel dephasing as a function of voxel size
- Measured velocity bias across different flow profiles (plug, parabolic, blunted)
- Velocity noise and the VENC trade-off
- Gradient hardware constraints (amplitude and slew rate limits)
- A 2D beating heart model with pulsatile flow and PC-MRI phase encoding

---

## Repository Structure

```
├── README.md
├── requirements.txt
├── bipolar_gradient.py     # Figure 1: bipolar gradient waveform and phase accumulation
├── functions.py            # Figures 2–6: all simulation studies
└── report.pdf              # Full report
```

---

## Requirements

Python 3.8 or later is required. Install dependencies with:

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `matplotlib`

---

## Usage

### Bipolar gradient figure

```bash
python bipolar_gradient.py
```

Saves `bipolar_gradient.pdf` and `bipolar_gradient.png`.

### Simulation figures

Run all simulation figures at once:

```bash
python - <<'EOF'
from functions import (
    plot_voxel_dephasing_vs_size,
    plot_flow_profile_effect,
    plot_venc_optimisation,
    plot_gradient_hardware_limits,
    plot_beating_heart_example,
)
plot_voxel_dephasing_vs_size()
plot_flow_profile_effect()
plot_venc_optimisation()
plot_gradient_hardware_limits()
plot_beating_heart_example()
EOF
```

Or call any function individually, for example:

```bash
python -c "from functions import plot_beating_heart_example; plot_beating_heart_example()"
```

Each function displays the figure and saves the corresponding PNG to the working directory:

| Function | Output file(s) |
|---|---|
| `plot_voxel_dephasing_vs_size()` | `fig_voxel_dephasing.png` |
| `plot_flow_profile_effect()` | `fig_flow_profiles.png`, `fig_signal_loss_profiles.png` |
| `plot_venc_optimisation()` | `fig_venc_noise.png`, `fig_aliasing.png` |
| `plot_gradient_hardware_limits()` | `fig_venc_hardware.png`, `fig_te_hardware.png` |
| `plot_beating_heart_example()` | `fig_heart_velocity.png`, `fig_heart_phase.png` |

---

## Background

Phase contrast MRI encodes blood flow velocity into the MRI signal phase using bipolar magnetic field gradients. A spin moving at velocity $v$ through a bipolar gradient of amplitude $G$, lobe duration $\delta$, and lobe separation $\Delta$ accumulates a phase

$$\phi = \gamma G \delta \Delta v$$

where $\gamma = 2.675 \times 10^8$ rad s$^{-1}$ T$^{-1}$ is the proton gyromagnetic ratio. By acquiring two datasets with opposite gradient polarities and subtracting, background phase is cancelled and velocity is extracted at each voxel. The maximum measurable velocity without aliasing is set by the velocity encoding parameter (VENC):

$$\text{VENC} = \frac{\pi}{\gamma G \delta \Delta}$$

For full derivations and theoretical background, refer to the accompanying report.

---

## Report

The full report is available in this repository as `report.pdf`.

---

## License

This code is made available for academic and educational purposes. If you use or adapt it, please cite the accompanying report.
