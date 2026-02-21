# ParaFit — Bleaney Model Fitting GUI
![version](https://img.shields.io/badge/version-0.2.0-blue) ![license](https://img.shields.io/badge/license-BSD%203--Clause-green)

This repository provides an interactive GUI to fit variable-temperature paramagnetic NMR shifts to the **extended Bleaney model** with **variable projection** for per-nucleus contact factors $F_i$ and shared global parameters $(S_1,S_2,D_2,D_3)$. It also exports diagnostics, linear approximations, and $Δχ_ax(T)$ in multiple unit conventions.

---

## Model

Fitting for each nucleus *i*:

$$
\delta_i(T) = F_i\left(\frac{S_1}{T}+\frac{S_2}{T^2}\right)
            + G_i\left(\frac{D_2}{T^2}+\frac{D_3}{T^3}\right)
$$

- **Globals (shared):** S₁, S₂, D₂, D₃  
- **Per nucleus:** Fᵢ (solved analytically each iteration, variable projection)  
- Ridge regularization $(\lambda)$ on Fᵢ is available.

**Baseline for comparison (classical Bleaney):**  
$S_2 = 0, D_3 = 0$

**Linear two-term approximation (per nucleus):**  
$\delta \approx \frac{a}{T} + \frac{b}{T^2}$

so that $(D_{2,i} \approx b/G_i)$ when $(S_2\approx 0, D_3\approx 0)$.

---

### Units and Scaling

Input $(G_i)$ may be given in Å⁻³. It is internally rescaled to SI units:

$$
G_i^{(\mathrm{SI})} = G_i^{(\mathrm{Å^{-3}})} \times \frac{10^{30}}{12\pi}
$$

so that the PCS term becomes  
$( G_i^{(\mathrm{SI})} (D_2/T^2 + D_3/T^3) )$ with $(X(T))$ in ppm.

The temperature-dependent axial susceptibility anisotropy is:

$$
\Delta\chi_{\mathrm{ax}}(T) = 12\pi \left( \frac{D_2}{T^2} + \frac{D_3}{T^3} \right) \times 10^{-6}
$$

It can be converted to **m³/mol**:
- Without μ₀ (legacy convention)
- With μ₀ (strict SI definition)

Both are exported.

---

## Features

- Global nonlinear least-squares with **variable projection** for Fᵢ  
- Optional **ridge** penalty λ on Fᵢ  
- **λ-sweep** tool with plots & CSV summary  
- Optional **PCS-guess prior** at defined T_ref  
- **Linear two-term WLS** and **Cartesian OLS** (PCS vs Gᵢ)  
- Exports **CSV/JSON/plot**  
- $Δχ_{ax}(T)$ plotting and export (extended, baseline, linear)

---

## Installation

### Requirements
- Python 3.10+
- PyQt6, NumPy, SciPy, Pandas, Matplotlib
