# Phase-Contrast MRI Simulation

Supplementary Python code for phase-contrast MRI (PC-MRI) simulation studies.

**Authors:** Mingyu Liang, Zak Nabi, Jamie Ward  
**Supervisor:** Dr. Andrew Scott

## Overview

This repository contains simulation code for:

- Bipolar gradient encoding and phase accumulation
- Intravoxel dephasing vs voxel size
- Flow-profile effects (plug, parabolic, blunted, boundary-layer)
- Bland-Altman agreement analysis for noisy reconstructed velocity
- Velocity uncertainty vs position and vs VENC
- Gradient hardware constraint sweeps (VENC, TE proxy, uncertainty)
- A 2D beating-heart velocity and phase-encoding model

## Repository Structure

```text
├── README.md
├── requirements.txt
├── bipolar_gradient.py   # standalone bipolar gradient waveform/phase figure
├── functions.py          # simulation functions and plotting entry points
└── bland_altman.py       # renamed from BA_Graph_Plot.py; Bland-Altman analysis script
```

## Requirements

Python 3.8+:

```bash
pip install -r requirements.txt
```

Dependencies:

- `numpy`
- `matplotlib`

## Usage

Run the standalone bipolar gradient figure:

```bash
python bipolar_gradient.py
```

Run the Bland-Altman analysis script:

```bash
python bland_altman.py
```

Run selected simulation figures from `functions.py`:

```bash
python - <<'EOF'
from functions import (
    plot_voxel_dephasing_vs_size,
    plot_flow_profile_effect,
    plot_plug_boundary_layer_velocity,
    plot_plug_boundary_layer_signal_magnitude,
    plot_velocity_uncertainty_vs_position,
    plot_venc_uncertainty_boundary_layer,
    plot_venc_optimisation,
    plot_gradient_hardware_limits,
    plot_beating_heart_example,
)

plot_voxel_dephasing_vs_size()
plot_flow_profile_effect()
plot_plug_boundary_layer_velocity()
plot_plug_boundary_layer_signal_magnitude()
plot_velocity_uncertainty_vs_position()
plot_venc_uncertainty_boundary_layer()
plot_venc_optimisation()
plot_gradient_hardware_limits()
plot_beating_heart_example()
EOF
```

The plotting functions in `functions.py` display figures but do not save files by default.
