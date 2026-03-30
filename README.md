# Simulating Motion-Encoded MRI for Quantifying Blood Flow and Diffusion in the Cardiovascular System

Supplementary code for the report produced for the **Physics of Medical Imaging and Radiotherapy** course at **Imperial College London**, April 2026.

**Authors:** Mingyu Liang, Zak Nabi, Jamie Ward  
**Supervisor:** Dr. Andrew Scott

---

## Overview

This repository contains the Python simulation framework developed to investigate motion-encoded MRI under realistic cardiovascular conditions. The code models phase contrast MRI signal formation from blood flowing through vessels and cardiac chambers, allowing the effects of key acquisition parameters to be studied under controlled conditions where the ground-truth velocity field is known exactly.

The simulations cover:

- Bipolar gradient encoding and phase accumulation
- Intravoxel dephasing as a function of voxel size and flow profile
- Velocity noise and the VENC trade-off
- Gradient hardware constraints (amplitude and slew rate limits)
- A 2D beating heart model with pulsatile flow

---

## Repository Structure

```
├── README.md
├── requirements.txt
└── [scripts and modules to be added]
```

---

## Requirements

Python 3.8 or later is required. Install dependencies with:

```bash
pip install -r requirements.txt
```

Key dependencies:

- `numpy`
- `matplotlib`
- `scipy`

---

## Usage

Instructions for running the simulations and reproducing the figures in the report will be added here as scripts are uploaded.

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
