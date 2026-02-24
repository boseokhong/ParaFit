# ParaFit — Bleaney Model Fitting GUI
![version](https://img.shields.io/badge/version-0.2.0-blue) ![license](https://img.shields.io/badge/license-BSD%203--Clause-green) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18752246.svg)](https://doi.org/10.5281/zenodo.18752246)



<img width="1203" height="762" alt="image" src="https://github.com/user-attachments/assets/057ad09e-785b-4cc0-bc6d-713cf6146c3c" />
ParaFit is a GUI application for analyzing temperature-dependent paramagnetic NMR shifts using Bleaney-type models, including linear, classical (baseline), and extended global fitting approaches.

It implements a global nonlinear fitting framework with **variable projection** for per-nucleus contact factors and shared magnetic susceptibility parameters, together with diagnostic exports and susceptibility reconstruction.

---
## Requirements

Python 3.10+ recommended.
> [!NOTE]
> The code requires `PyQt6`, `numpy`, `pandas`, `scipy`, `matplotlib` packages to run.

---
## Running

python main.py

---
## Model

For each nucleus *i*:

$$
\delta_i(T) =
F_i\left(\frac{S_1}{T}+\frac{S_2}{T^2}\right)+
G_i\left(\frac{D_2}{T^2}+\frac{D_3}{T^3}\right)
$$

- **Global parameters (shared across nuclei):**  
  $S_1$, $S_2$, $D_2$, $D_3$

- **Per-nucleus parameter:**  
  $F_i$ (solved analytically at each iteration via variable projection)

- Optional **ridge regularization ($\lambda$)** can be applied to stabilize $F_i$.

---

## Model Hierarchy

ParaFit supports three levels of analysis:

### Extended Bleaney model
Full model with $S_1, S_2, D_2, D_3$ fitted globally.

### Classical (baseline) Bleaney limit
$$
S_2 = 0, \quad D_3 = 0
$$

### Linear two-term approximation (per nucleus)
$$
\delta(T) \approx \frac{a}{T} + \frac{b}{T^2}
$$

which yields approximately

$$
D_{2,i} \approx \frac{b}{G_i}\quad(S_2 \approx 0,\; D_3 \approx 0)
$$

This hierarchy enables consistent comparison between phenomenological linear analysis and physically constrained global models.

---

## Units and Scaling

The geometrical factor $G_i$ may be provided in Å⁻³.  
It is internally rescaled to SI-compatible units as:

$$
G_i^{(\mathrm{SI})}=G_i^{(\mathrm{Å^{-3}})}\times\frac{10^{30}}{12\pi}
$$

The temperature-dependent axial susceptibility anisotropy is reconstructed as:

$$
\Delta\chi_{\mathrm{ax}}(T)=12\pi\left(\frac{D_2}{T^2}+\frac{D_3}{T^3}\right)\times 10^{-6}
$$

Exports include:

- Dimensionless per-molecule values  
- m³/mol (without $\mu_0$; legacy convention)  
- m³/mol (with $\mu_0$; strict SI definition)

---
## Intended Use

ParaFit is designed for
- Temperature-dependent paramagnetic NMR analysis 
- PCS/FCS separation studies 
- $Δχ_{ax}$ extraction 
- Comparing classical and extended Bleaney models 
- Integrating ORCA χ(T) results into pNMR analysis

---

## Features

### 1. CSV-driven workflow

Load any CSV file and map columns in the GUI.

- Required columns : nucleus, T (K), delta_para (ppm)

- Optional columns : $G_i$, weight (default 1.0), PCS_guess (ppm)

If $G_i$ is not provided, the program automatically switches to linear-only mode (no baseline/extended global fit).

---

### 2. Linear two-term regression (per nucleus)

Model: $δ(T) ≈ a/T + b/T^2$

-   No-intercept weighted least squares
-   Reports:
    -   a, b and standard errors
    -   FCS and PCS at user-defined $T_{ref}$
    -   If $G_i$ is available: $D_{2,i} = b / G_i$
-   Computes weighted mean $D_2$ when uncertainties are available.

---

### 3. Classical (Baseline) Bleaney model

Optional baseline fit with constraints: $S_2 = 0$, $D_3 = 0$\
Useful for comparing against the extended model.

---

### 4. Extended global fit

Simultaneous nonlinear least-squares fit of: 
- $S_1$, $S_2$, $D_2$, $D_3$ (global) 
- $F_i$ (per nucleus, solved analytically each iteration)

Outputs: 
- Global parameters 
- Overall RMSE and SSE 
- Per-nucleus diagnostics: 
	- number of data points 
	- temperature range 
	- RMSE 
	- $F_i $
	- FCS/PCS at $T_{ref}$

Also computes: 
- $τ_1 = S_2/S_1$
- $τ_2 = D_3/D_2$ 
- normalized $τ$ metrics at $T_{ref}$

---

### 5. Ridge regularization on $F_i$

A ridge parameter $λ$ can be applied to stabilize $F_i$: 
- Adds $√λ · F_i$ to residual vector 
- Useful when $F_i$ values become unstable

Includes a Ridge $λ$ sweep tool: 
- Log-spaced $λ$ scan 
- Plots: 
	- mean FCS/PCS vs $λ$
	- $D_2, D_3$ vs $λ$
	- RMSE vs $λ$ 
	- Exportable as CSV

---

### 6. ORCA χ(T) integration

`Load ORCA .out files (χ mode)`
- Parses temperature-dependent susceptibility tensors 
- Converts to χ(T) in SI units ($m^3$ per molecule) 
- Computes $Δχ_{ax}$ and $Δχ_{rh}$
- Can attach ORCA-based PCS guess column to CSV 
- Can estimate initial $D_2$ and $D_3$ from $Δχ_{ax}(T)$

> [!NOTE]
>⚠️ This module is currently being expanded. Additional functionality will be added soon.


---

### 7. ORCA AILFT parsing

`Load ORCA .out (AILFT mode)` extracts: 
- Configuration 
- Shell type 
- CI blocks 
- Active MO range 
- SOC constants 
- Slater--Condon parameters 
- Racah parameters 
- Ligand field eigenvalues 
- VLFT matrix (if available)

> [!NOTE]
>⚠️ This module is currently being expanded. Additional functionality will be added soon.

---
## Input CSV Format

Minimal example structure:
```
nucleus,T_K,delta_para_ppm,G_i,weight,PCS_guess_ppm
H3,298,1.23,2.1,1.0,
H3,308,1.11,2.1,1.0,
H5,298,0.87,1.3,1.0,
```
> [!NOTE]
>- Temperatures must be in Kelvin. 
>- $G_i$ is assumed to be in $Å^{-3}$. 
>- Internally scaled to SI via: $G_{scaled} = G_{raw} × (10^{30} / 12π)$

The GUI can generate a template via `Save Sample CSV`.

---

## Outputs

`Save JSON` exports 
- Baseline results (if run) 
- Extended results 
- Global parameters 
- τ metrics 
- Fi values 
- Diagnostics 
- Overall metrics

---

`Save Plots` creates 
- Per-nucleus data vs model plots 
- Residual plots 
- Linear δ vs 1/T plots 
- Baseline and extended folders separated

```
output_folder/
├── extended_plots/
│   ├── H1_fit.png
│   ├── H1_residuals.png
│   ├── H2_fit.png
│   └── ...
├── baseline_plots/      (only if baseline was run)
│   ├── H1_fit.png
│   ├── H1_residuals.png
│   └── ...
└── linear_plots/        (if linear regression enabled)
    ├── H1_delta_vs_invT.png
    ├── H2_delta_vs_invT.png
    └── ...
```

---

`Save Data (CSV)` exports structured folder:

```
output_folder/
├── linear/
│   ├── linear_approx_table.csv
│   ├── linear_approx_weighted_summary.csv
│   ├── cartesian_ols_summary.csv
│   ├── cartesian_points_PCS_vs_Gi.csv
│   └── linear_approx_table_for_gui.csv
│
├── extended/
│   ├── globals_and_metrics.csv
│   ├── summary_per_nucleus.csv
│   └── per_nucleus/
│       ├── H1_extended.csv
│       ├── H2_extended.csv
│       └── ...
│
└── baseline/                 (only if baseline was run)
    ├── globals_and_metrics.csv
    ├── summary_per_nucleus.csv
    └── per_nucleus/
        ├── H1_baseline.csv
        ├── H2_baseline.csv
        └── ...
```
Includes 
- Per-nucleus model breakdown 
- Contact and pseudocontact contributions 
- Global summaries 
- Cartesian PCS vs $G_i$ OLS results

---

`Save Δχ_ax vs T` exports temperature grid tables for: 

- Extended model ($D_2$, $D_3$) 
- Baseline model ($D_2$ only) 
- Linear approximation
```
output_folder/
├── DeltaChi_ax_vs_T_extended.csv
├── DeltaChi_ax_vs_T_baseline.csv
└── DeltaChi_ax_vs_T_linear_approx.csv
```
Includes
- Per molecule (dimensionless) 
- m³/mol (no μ0) 
- m³/mol (with μ0)

---

## Project Structure
```
main.py
logic/
├─ orca_susc.py
├─ orca_AILFT.py
├─ orca_viewer.py
└─ orca_parse_core.py
```
---
## About
This project was created by **Boseok Hong** (**Department of Chemistry of the f-elements, Institute of Resource Ecology, HZDR**).  
For inquiries, please contact [bshong66@gmail.com](mailto:bshong66@gmail.com) or [b.hong@hzdr.de](mailto:b.hong@hzdr.de).

2026.2. Boseok Hong  

---
