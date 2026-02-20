# logic/orca_susc.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import scipy.constants as const

from .orca_parse_core import scan_temp_tagged_tensor_blocks


# ORCA block markers
_ORCA_TEMP = re.compile(r"TEMPERATURE/K:\s*([0-9]+(?:\.[0-9]+)?)")
_ORCA_TENSOR_HEADER = re.compile(r"Tensor in molecular frame\s*\(cm3\*K/mol\)\s*:", re.IGNORECASE)


# ----------------------------
# Data model
# ----------------------------
@dataclass
class ChiTensorSeries:
    """
    Temperature-dependent susceptibility tensors.

    tensors_per_molecule_m3: dict[T(K)] -> 3x3 numpy array in SI (m^3/molecule)
      - stored as chi(T) (NOT chi*T)
    is_chiT_input: whether original parsed data were chi*T
    used_4pi: whether 4π factor was applied in conversion
    """
    tensors_per_molecule_m3: Dict[float, np.ndarray]
    is_chiT_input: bool = True
    used_4pi: bool = True


# ----------------------------
# Unit helpers
# ----------------------------
def chiT_cm3Kmol_to_chi_per_molecule_m3(chiT_cm3Kmol: float, T: float, *, use_4pi: bool = True) -> float:
    """
    Convert molar susceptibility*temperature in (cm^3*K/mol) into chi(T) in SI per molecule (m^3/molecule).

    Steps (Spinach-style):
      chi_cm3_per_mol = chiT_cm3Kmol / T
      chi_m3_per_mol  = chi_cm3_per_mol * 1e-6
      if use_4pi: chi_m3_per_mol *= 4π
      chi_m3_per_molecule = chi_m3_per_mol / N_A
    """
    chi_cm3_per_mol = float(chiT_cm3Kmol) / float(T)
    chi_m3_per_mol = chi_cm3_per_mol * 1e-6
    if use_4pi:
        chi_m3_per_mol *= (4.0 * np.pi)
    chi_m3_per_molecule = chi_m3_per_mol / const.N_A
    return float(chi_m3_per_molecule)

# ----------------------------
# ORCA parser (chiT tensor blocks)
# ----------------------------
def read_orca_temp_dependent_chiT_tensor(text: str, *, use_4pi: bool = True) -> ChiTensorSeries:
    """
    Parse ORCA output blocks like:

    TEMPERATURE/K:     298.00
    Tensor in molecular frame (cm3*K/mol):
        a11 a12 a13
        a21 a22 a23
        a31 a32 a33

    ORCA reports chi*T in (cm^3*K/mol). We convert to chi(T) as SI per molecule (m^3/molecule)
    """
    # scan_temp_tagged_tensor_blocks (your core) returns chi(T) in SI (m^3/mol)
    tensors_m3_per_mol = scan_temp_tagged_tensor_blocks(
        text,
        temp_regex=_ORCA_TEMP,
        header_regex=_ORCA_TENSOR_HEADER,
        header_stop_regexes=[_ORCA_TEMP],
        use_4pi=use_4pi,
    )
    if not tensors_m3_per_mol:
        raise ValueError("No ORCA temperature-dependent susceptibility tensor blocks were found.")

    # convert to per-molecule (m^3/molecule)
    tensors_m3_per_molecule: Dict[float, np.ndarray] = {}
    for T, chi_m3mol in tensors_m3_per_mol.items():
        tensors_m3_per_molecule[float(T)] = np.asarray(chi_m3mol, float) / const.N_A

    return ChiTensorSeries(
        tensors_per_molecule_m3=tensors_m3_per_molecule,
        is_chiT_input=True,
        used_4pi=use_4pi,
    )

# ----------------------------
# Tensor helpers
# ----------------------------
def make_traceless(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float).reshape(3, 3)
    iso = np.trace(t) / 3.0
    return t - np.eye(3) * iso

def mehring_order_from_tensor(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize tensor and return (eigvals_sorted, eigvecs_sorted) in Mehring-like order:
    choose z as the axis with largest |eigenvalue| (typically after tracelessness).

    Returns:
      eigvals_sorted: [chi_xx, chi_yy, chi_zz]
      eigvecs_sorted: 3x3, columns correspond to x,y,z eigenvectors in same order.
    """
    t = np.asarray(t, float).reshape(3, 3)
    vals, vecs = np.linalg.eigh(t)  # ascending

    z_idx = int(np.argmax(np.abs(vals)))
    rest = [i for i in range(3) if i != z_idx]

    # deterministic x/y ordering: yy has larger |val| among remaining
    if abs(vals[rest[0]]) >= abs(vals[rest[1]]):
        y_idx, x_idx = rest[0], rest[1]
    else:
        y_idx, x_idx = rest[1], rest[0]

    order = [x_idx, y_idx, z_idx]
    return vals[order], vecs[:, order]

def delta_chi_ax_rh_from_principal(chi_xx: float, chi_yy: float, chi_zz: float) -> Tuple[float, float]:
    """
    Conventions:
      Delta_chi_ax = chi_zz - 1/2(chi_xx + chi_yy)
      Delta_chi_rh = chi_xx - chi_yy
    Units follow input (here: m^3/molecule).
    """
    d_ax = float(chi_zz - 0.5 * (chi_xx + chi_yy))
    d_rh = float(chi_xx - chi_yy)
    return d_ax, d_rh

def estimate_D2_D3_from_orca_chi_ax(series, *, traceless=True, mehring=True):
    # 1) build arrays
    Ts = np.array(sorted(series.tensors_per_molecule_m3.keys()), float)
    y = []
    for T in Ts:
        chi = interpolate_tensor(series, float(T))
        if traceless:
            chi = make_traceless(chi)
        if mehring:
            vals, _ = mehring_order_from_tensor(chi)
            cxx, cyy, czz = map(float, vals)
        else:
            cxx, cyy, czz = float(chi[0,0]), float(chi[1,1]), float(chi[2,2])
        dax, _ = delta_chi_ax_rh_from_principal(cxx, cyy, czz)  # m^3/molecule
        y.append(dax)
    y = np.asarray(y, float)

    # 2) fit Δχ(T) = A/T^2 + B/T^3  (A,B in m^3*K^2/molecule, m^3*K^3/molecule)
    X = np.column_stack([1.0/(Ts**2), 1.0/(Ts**3)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    A, B = float(beta[0]), float(beta[1])

    # 3) convert to your model's D2/D3 (ppm-scaled coefficients when G is SI-scaled)
    D2 = 1e6 * A
    D3 = 1e6 * B

    # RMSE in Δχ-space (m^3/molecule)
    rmse = float(np.sqrt(np.mean((y - X @ beta)**2)))
    return D2, D3, rmse

# ----------------------------
# Parsing helper for pre-extracted principal table
# ----------------------------
def read_chiT_principal_table(
    text: str,
    *,
    use_4pi: bool = True,
) -> ChiTensorSeries:
    """
    Read a whitespace table of principal components reported as chiT in (cm^3*K/mol):
      T   chiT_xx   chiT_yy   chiT_zz

    Returns tensors as chi(T) in SI per molecule (m^3/molecule), diagonal in that principal frame.
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        T = float(parts[0])
        cxx = float(parts[1])
        cyy = float(parts[2])
        czz = float(parts[3])
        rows.append((T, cxx, cyy, czz))

    tensors: Dict[float, np.ndarray] = {}
    for T, cxx, cyy, czz in rows:
        xx = chiT_cm3Kmol_to_chi_per_molecule_m3(cxx, T, use_4pi=use_4pi)
        yy = chiT_cm3Kmol_to_chi_per_molecule_m3(cyy, T, use_4pi=use_4pi)
        zz = chiT_cm3Kmol_to_chi_per_molecule_m3(czz, T, use_4pi=use_4pi)
        tensors[float(T)] = np.diag([xx, yy, zz])

    if not tensors:
        raise ValueError("No rows were parsed from principal chiT table.")

    return ChiTensorSeries(
        tensors_per_molecule_m3=tensors,
        is_chiT_input=True,
        used_4pi=use_4pi,
    )

# ----------------------------
# Interpolation
# ----------------------------
def interpolate_tensor(series: ChiTensorSeries, T: float) -> np.ndarray:
    """
    Linear interpolation of tensors (component-wise) in SI per-molecule units.
    If outside range, clamps to nearest endpoint.
    """
    Ts = np.array(sorted(series.tensors_per_molecule_m3.keys()), float)
    if Ts.size == 0:
        raise ValueError("Empty ChiTensorSeries.")

    if T <= Ts[0]:
        return np.array(series.tensors_per_molecule_m3[float(Ts[0])], float)
    if T >= Ts[-1]:
        return np.array(series.tensors_per_molecule_m3[float(Ts[-1])], float)

    hi = int(np.searchsorted(Ts, T))
    lo = hi - 1
    T0, T1 = float(Ts[lo]), float(Ts[hi])
    w = (T - T0) / (T1 - T0)

    A = np.asarray(series.tensors_per_molecule_m3[T0], float)
    B = np.asarray(series.tensors_per_molecule_m3[T1], float)
    return (1.0 - w) * A + w * B

# ----------------------------
# PCS prediction using existing G columns (Gi_raw = Å^-3)
# ----------------------------
def attach_pcs_from_chi(
    df: pd.DataFrame,
    chi_series: ChiTensorSeries,
    *,
    T_col: str = "T",
    Gax_raw_col: str = "G_i_raw",         # Å^-3, NO 1/(12π) included
    Grh_raw_col: Optional[str] = None,    # Å^-3, optional
    out_col: str = "PCS_guess_orca_ppm",
    traceless: bool = True,
    mehring: bool = True,
    include_12pi: bool = True,
) -> pd.DataFrame:
    """
    Attach PCS (ppm) predicted from ORCA chi(T).

    Assumptions (your convention):
      - chi_series tensors are chi(T) in m^3/molecule
      - Gax_raw_col is Gi_raw in Å^-3 (i.e., r in Å; and WITHOUT 1/(12π) factor)
      - PCS axial contribution:
            PCS_ppm = (Gi_raw/(12π)) * Δχ_ax(Å^3) * 1e6
        with Δχ_ax(Å^3) = Δχ_ax(m^3) * 1e30
        => PCS_ppm = Gi_raw * Δχ_ax(m^3/molecule) * 1e36 / (12π)

      - If rhombic term provided:
            + (Grh_raw/(12π)) * Δχ_rh(Å^3) * 1e6
    """
    out = df.copy()

    if T_col not in out.columns:
        raise KeyError(f"Missing column '{T_col}' in dataframe.")
    if Gax_raw_col not in out.columns:
        raise KeyError(f"Missing column '{Gax_raw_col}' in dataframe.")
    if Grh_raw_col and (Grh_raw_col not in out.columns):
        raise KeyError(f"Missing column '{Grh_raw_col}' in dataframe.")

    Ts = out[T_col].to_numpy(float)
    Gax = out[Gax_raw_col].to_numpy(float)
    Grh = out[Grh_raw_col].to_numpy(float) if Grh_raw_col else None

    pcs = np.full(len(out), np.nan, float)

    # factor that maps: (Å^-3)*(m^3) -> ppm
    # ppm = Gi_raw * dchi(m^3) * 1e36 / (12π)   (if include_12pi)
    base = 1e36
    if include_12pi:
        base = base / (12.0 * np.pi)

    for i in range(len(out)):
        T = float(Ts[i])
        gi = float(Gax[i])
        if not (np.isfinite(T) and np.isfinite(gi)):
            continue

        chi = interpolate_tensor(chi_series, T)  # m^3/molecule
        if traceless:
            chi = make_traceless(chi)

        if mehring:
            vals, _vecs = mehring_order_from_tensor(chi)
            chi_xx, chi_yy, chi_zz = map(float, vals)
        else:
            chi_xx, chi_yy, chi_zz = float(chi[0, 0]), float(chi[1, 1]), float(chi[2, 2])

        d_ax_m3, d_rh_m3 = delta_chi_ax_rh_from_principal(chi_xx, chi_yy, chi_zz)

        pcs_val = gi * d_ax_m3
        if Grh is not None:
            grh = float(Grh[i])
            if np.isfinite(grh):
                pcs_val += grh * d_rh_m3

        pcs[i] = float(pcs_val) * base

    out[out_col] = pcs
    return out
