"""
Extended Bleaney Method Global Fitting GUI
Model:
  delta_i(T) = Fi*(S1/T + S2/T^2) + Gi*(D2/T^2 + D3/T^3)
- Globals (shared across nuclei): S1, S2, D2, D3
- Per nucleus: Fi, solved analytically each iteration (variable projection)
"""

from __future__ import annotations
import sys, math, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import re

from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PyQt6.QtGui import QAction, QKeySequence, QGuiApplication
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox, QLineEdit,
    QPushButton, QGroupBox, QCheckBox, QSplitter, QTableView, QTextEdit, QInputDialog,
    QAbstractItemView, QMenu, QSizePolicy, QGridLayout
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import scipy.constants as const

from logic.orca_viewer import OrcaChiViewer, OrcaViewerInputs
from logic.orca_susc import (
    read_orca_temp_dependent_chiT_tensor,
    interpolate_tensor,
    make_traceless,
    mehring_order_from_tensor,
    delta_chi_ax_rh_from_principal,
    attach_pcs_from_chi,
    estimate_D2_D3_from_orca_chi_ax,
)
from logic.orca_AILFT import parse_orca_ailft

# -----------------------------
# Data structures and fitting core
# -----------------------------
@dataclass
class Dataset:
    groups: Dict[str, pd.DataFrame]
    ridge_lambda: float = 0.0
    has_G: bool = True  # Gi data available?

_nat_re = re.compile(r'(\d+)')
def natural_key(s) -> tuple:
    # "H1", "H11", "C13a" 같은 라벨을 ["h", 1, "", ""] 식으로 분해해서 정렬
    parts = _nat_re.split(str(s))
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)

def prepare_dataset(df: pd.DataFrame,
                    col_nuc: str, col_T: str, col_delta: str,
                    col_G: Optional[str] = None,
                    col_w: Optional[str] = None,
                    col_pcs_guess: Optional[str] = None) -> Dataset:
    """
    Gi 없이도 linear approximation이 가능하도록 Gi를 옵션으로 처리.
    - 필수 컬럼: nucleus, T, delta
    - 선택 컬럼: G, weight, PCS_guess
    - 반환: Dataset(groups=..., has_G=bool)
    """
    # --- 필수 컬럼 점검 ---
    req = [col_nuc, col_T, col_delta]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in CSV.")

    # --- Gi 유무 판단 ---
    has_G = (col_G is not None) and (col_G in df.columns)

    # --- weight 컬럼 처리 ---
    if col_w is not None and col_w not in df.columns:
        col_w = None

    df = df.copy()

    # 선택 컬럼 포함 목록 구성 (Gi/guess는 없어도 진행)
    cols = req \
         + ([col_G] if has_G else []) \
         + ([col_w] if col_w else []) \
         + ([col_pcs_guess] if (col_pcs_guess and col_pcs_guess in df.columns) else [])

    # 필수 컬럼만 NaN 제거 (Gi/guess는 NaN 허용)
    df = df[cols].dropna(subset=[col_T, col_delta])

    # 온도 sanity check
    if (df[col_T] < 1).any():
        raise ValueError("Temperatures must be in Kelvin (found < 1 K).")

    # weight 없으면 1.0 부여
    if col_w is None:
        df["_w_"] = 1.0
        col_w = "_w_"
    else:
        df[col_w] = df[col_w].astype(float)
        if (df[col_w] <= 0).any():
            raise ValueError("All weights must be > 0.")

    # 정렬
    df = df.sort_values([col_nuc, col_T]).reset_index(drop=True)

    # --- nucleus별 그룹 구성 ---
    groups: Dict[str, pd.DataFrame] = {}
    for key, sub in df.groupby(col_nuc):
        # 기본 rename (필수 3개 + weight)
        renamed = sub.rename(columns={col_T: "T", col_delta: "delta", col_w: "w"})

        # Gi가 있으면 G→"G", 그리고 "G_raw" 보존
        if has_G:
            renamed = renamed.rename(columns={col_G: "G"})
            renamed["G_raw"] = renamed["G"].astype(float)
        else:
            # Gi가 없으면 G_raw만 NaN으로 생성 (downstream 호환)
            renamed["G_raw"] = np.nan

        # PCS guess가 있으면 부착
        if col_pcs_guess and (col_pcs_guess in sub.columns):
            renamed["pcs_guess"] = sub[col_pcs_guess].astype(float)

        groups[str(key)] = renamed

    groups = {k: groups[k] for k in sorted(groups.keys(), key=natural_key)}

    # Dataset에 has_G 플래그를 담아 반환 (Dataset 정의에 has_G: bool 필드가 있어야 함)
    return Dataset(groups=groups, ridge_lambda=0.0, has_G=has_G)

def solve_Fi_for_group(sub: pd.DataFrame,
                       S1: float, S2: float, D2: float, D3: float,
                       ridge_lambda: float = 0.0) -> float:
    # Weighted ridge LS in closed form for single nucleus group
    T = sub["T"].to_numpy(float)
    d = sub["delta"].to_numpy(float)
    G = sub["G"].to_numpy(float)
    w = sub["w"].to_numpy(float)

    x = (S1 / T) + (S2 / (T**2))            # contact regressor
    pc = (D2 / (T**2)) + (D3 / (T**3))      # pseudocontact scalar
    y = d - G * pc                          # target after removing pc part

    wx2 = float(np.sum(w * x * x))
    wxy = float(np.sum(w * x * y))
    if wx2 == 0.0 and ridge_lambda == 0.0:
        return 0.0

    if ridge_lambda > 0.0:
        return wxy / (wx2 + float(ridge_lambda))
    return wxy / wx2

# -----lambda swip util start --------
def sweep_ridge_lambda(ds: Dataset, init: dict, fix: dict,
                       lambdas: list[float], Tref: float = 298.0):
    """
    여러 λ(=alpha) 값에 대해 Extended 모델을 반복 적합하고 요약 테이블을 반환.
    반환 컬럼:
      - lambda, S1,S2,D2,D3, RMSE
      - FCS_mean_ppm, PCS_mean_ppm
      - FCS_median_ppm, PCS_median_ppm
      - FCS_absmean_ppm, PCS_absmean_ppm
      - FCS_range_ppm, PCS_range_ppm
    """
    def _stats(vals):
        import numpy as _np
        a = _np.asarray(vals, float)
        a = a[_np.isfinite(a)]
        if a.size == 0:
            return float("nan"), float("nan"), float("nan"), float("nan")  # mean, median, absmean, range
        mean   = float(_np.mean(a))
        median = float(_np.median(a))
        absmean = float(_np.mean(_np.abs(a)))
        rng    = float(_np.max(a) - _np.min(a))
        return mean, median, absmean, rng

    out_rows = []
    for lam in lambdas:
        ds.ridge_lambda = float(lam)
        try:
            globals_out, fi_map, diag = fit_globals(ds, init, fix, Tref_for_diag=Tref)
            met = overall_metrics(ds, globals_out, fi_map)

            fcs_vals = [v.get("fcs_at_Tref_ppm", float("nan")) for v in diag.values()]
            pcs_vals = [v.get("pcs_at_Tref_ppm", float("nan")) for v in diag.values()]

            fcs_mean, fcs_med, fcs_absmean, fcs_rng = _stats(fcs_vals)
            pcs_mean, pcs_med, pcs_absmean, pcs_rng = _stats(pcs_vals)

            out_rows.append({
                "lambda": lam,
                "S1": globals_out["S1"], "S2": globals_out["S2"],
                "D2": globals_out["D2"], "D3": globals_out["D3"],
                "RMSE": met["RMSE"],

                "FCS_mean_ppm": fcs_mean,
                "PCS_mean_ppm": pcs_mean,

                "FCS_median_ppm": fcs_med,
                "PCS_median_ppm": pcs_med,
                "FCS_absmean_ppm": fcs_absmean,
                "PCS_absmean_ppm": pcs_absmean,
                "FCS_range_ppm": fcs_rng,
                "PCS_range_ppm": pcs_rng,
            })
        except Exception as e:
            out_rows.append({"lambda": lam, "error": str(e)})

    return pd.DataFrame(out_rows)

class RidgeSweepWindow(QWidget):
    """λ 스윕 대화창."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ridge λ Sweep")
        self.resize(900, 600)
        lay = QVBoxLayout(self)

        # Matplotlib figure
        self.fig = Figure(figsize=(8.5, 4.2))
        self.ax_fpcs = self.fig.add_subplot(131)
        self.ax_dd   = self.fig.add_subplot(132)
        self.ax_rmse = self.fig.add_subplot(133)
        self.canvas = FigureCanvas(self.fig)
        lay.addWidget(self.canvas)

        # Table view and buttons
        bottom = QHBoxLayout()
        self.table = QTableView()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        bottom.addWidget(self.table, 1)

        btns = QVBoxLayout()
        self.btn_save_csv = QPushButton("Save CSV…")
        btns.addWidget(self.btn_save_csv)
        btns.addStretch(1)
        bottom.addLayout(btns, 0)

        lay.addLayout(bottom)

        # event handler
        self.btn_save_csv.clicked.connect(self._save_csv)
        self._last_df: Optional[pd.DataFrame] = None

    def set_results(self, df: pd.DataFrame):
        self._last_df = df.copy()
        # 테이블 모델
        class _DfModel(QAbstractTableModel):
            def __init__(self, df: pd.DataFrame): super().__init__(); self.df=df
            def rowCount(self, p=QModelIndex()): return len(self.df)
            def columnCount(self, p=QModelIndex()): return self.df.shape[1]
            def data(self, idx, role=Qt.ItemDataRole.DisplayRole):
                if not idx.isValid() or role != Qt.ItemDataRole.DisplayRole: return None
                val = self.df.iat[idx.row(), idx.column()]
                if isinstance(val, float): return f"{val:.6g}"
                return str(val)
            def headerData(self, sec, orient, role=Qt.ItemDataRole.DisplayRole):
                if role != Qt.ItemDataRole.DisplayRole: return None
                return self.df.columns[sec] if orient==Qt.Orientation.Horizontal else str(sec+1)
        self.table.setModel(_DfModel(self._last_df))

        # plot
        ax1, ax2, ax3 = self.ax_fpcs, self.ax_dd, self.ax_rmse
        for ax in (ax1, ax2, ax3): ax.clear()

        if "error" in df.columns:
            df = df[df["error"].isna()] if df["error"].notna().any() else df

        if not df.empty:
            # FCS/PCS mean vs lambda
            ax1.plot(df["lambda"], df["FCS_mean_ppm"], marker="o", label="FCS mean (ppm)")
            ax1.plot(df["lambda"], df["PCS_mean_ppm"], marker="s", label="PCS mean (ppm)")
            ax1.set_xscale("log"); ax1.set_xlabel("ridge λ (α)"); ax1.set_ylabel("ppm"); ax1.legend()
            ax1.set_title("FCS/PCS at T_ref (mean)")

            # D2, D3 vs lambda
            ax2.plot(df["lambda"], df["D2"], marker="o", label="D2")
            ax2.plot(df["lambda"], df["D3"], marker="s", label="D3")
            ax2.set_xscale("log"); ax2.set_xlabel("ridge λ (α)"); ax2.set_ylabel("value"); ax2.legend()
            ax2.set_title("Global Δχ parameters")

            # RMSE vs lambda
            ax3.plot(df["lambda"], df["RMSE"], marker="o")
            ax3.set_xscale("log"); ax3.set_xlabel("ridge λ (α)"); ax3.set_ylabel("RMSE (ppm)")
            ax3.set_title("Fit quality")

        self.fig.tight_layout()
        self.canvas.draw()

    def _save_csv(self):
        if self._last_df is None or self._last_df.empty:
            QMessageBox.information(self, "No data", "No sweep results to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save λ sweep CSV", "lambda_sweep.csv", "CSV (*.csv)")
        if not path: return
        try:
            self._last_df.to_csv(path, index=False, encoding="utf-8-sig")
            QMessageBox.information(self, "Saved", f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

# -----lambda swip util end--------

def residuals_global(theta: np.ndarray, ds: Dataset,
                     fix_mask: np.ndarray, fixed_vals: np.ndarray,
                     Tref_for_prior: Optional[float] = None,
                     lambda_pcs_prior: float = 0.0) -> np.ndarray:
    # rebuild full [S1,S2,D2,D3] from theta + fixed
    full = np.zeros(4, dtype=float)
    it = 0
    for j in range(4):
        if fix_mask[j]:
            full[j] = fixed_vals[j]
        else:
            full[j] = theta[it]; it += 1
    S1, S2, D2, D3 = full.tolist()
    res_list = []
    pcs_prior_terms = []

    for _, sub in ds.groups.items():
        Fi = solve_Fi_for_group(sub, S1, S2, D2, D3, ds.ridge_lambda)

        T = sub["T"].to_numpy(float)
        d = sub["delta"].to_numpy(float)
        G = sub["G"].to_numpy(float)
        w = sub["w"].to_numpy(float)

        model = Fi * (S1 / T + S2 / T ** 2) + G * (D2 / T ** 2 + D3 / T ** 3)
        res_list.append(np.sqrt(w) * (d - model))
        #r = np.sqrt(w) * (d - model)

        if ds.ridge_lambda > 0:
            res_list.append(np.array([math.sqrt(ds.ridge_lambda) * Fi], dtype=float))

        # --- PCS prior (있을 때만) ---
        if lambda_pcs_prior > 0.0 and "pcs_guess" in sub.columns:
            pcs_vec = sub["pcs_guess"].to_numpy(float)
            if np.isfinite(pcs_vec).any():
                pcs_guess = float(np.nanmean(pcs_vec))  # group 대표값
                # prior에 쓸 T_ref
                if Tref_for_prior is not None:
                    Tref = float(Tref_for_prior)
                else:
                    Tref = 298.0
                Gbar = float(np.mean(sub["G"]))  # already SI scale (1/(12π)·m^-3)
                pcs_model = Gbar * (D2 / (Tref ** 2) + D3 / (Tref ** 3))
                pcs_prior_terms.append(math.sqrt(lambda_pcs_prior) * (pcs_model - pcs_guess))

    if pcs_prior_terms:
        res_list.append(np.array(pcs_prior_terms, dtype=float))

    return np.concatenate(res_list)

def fit_globals(ds: Dataset,
                init: Dict[str, float],
                fix: Dict[str, float],
                Tref_for_diag: Optional[float] = None,
                Tref_for_prior: Optional[float] = None,
                lambda_pcs_prior: float = 0.0
                ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, dict]]:

    names = ["S1", "S2", "D2", "D3"]
    x0_full = np.array([init.get(n, 0.0) for n in names], dtype=float)
    fix_mask = np.array([n in fix for n in names], dtype=bool)
    fixed_vals = np.array([fix.get(n, 0.0) if fix_mask[i] else 0.0 for i, n in enumerate(names)], dtype=float)

    x0 = x0_full[~fix_mask]
    if x0.size == 0:
        raise ValueError("All global parameters are fixed; nothing to fit.")

    res = least_squares(
        fun=lambda th: residuals_global(th, ds, fix_mask, fixed_vals,
                                        Tref_for_prior=Tref_for_prior,
                                        lambda_pcs_prior=lambda_pcs_prior),
        x0=x0,
        method="trf",
        max_nfev=20000,
        verbose=0
    )

    full = np.zeros(4, dtype=float)
    it = 0
    for j in range(4):
        if fix_mask[j]:
            full[j] = fixed_vals[j]
        else:
            full[j] = res.x[it]; it += 1
    S1, S2, D2, D3 = full.tolist()

    Fi_map: Dict[str, float] = {}
    diag: Dict[str, dict] = {}
    for key in sorted(ds.groups.keys(), key=natural_key):
        sub = ds.groups[key]
        Fi = solve_Fi_for_group(sub, S1, S2, D2, D3, ds.ridge_lambda)
        Fi_map[key] = Fi

        T = sub["T"].to_numpy(float)
        d = sub["delta"].to_numpy(float)
        G = sub["G"].to_numpy(float)
        w = sub["w"].to_numpy(float)
        model = Fi * (S1/T + S2/(T**2)) + G * (D2/(T**2) + D3/(T**3))
        resid = d - model

        if Tref_for_diag is not None:
            Tref = float(Tref_for_diag)
        else:
            Tref = 298.0 if (T.min() <= 298 <= T.max()) else float(np.median(T))

        contact = Fi * (S1/Tref + S2/(Tref**2))
        pc = float(np.mean(G)) * (D2/(Tref**2) + D3/(Tref**3))
        diag[key] = {
            "n_points": int(len(T)),
            "T_min_K": float(T.min()),
            "T_max_K": float(T.max()),
            "RMSE_ppm": float(np.sqrt(np.average(resid**2, weights=w))),
            "fcs_at_Tref_ppm": float(contact),
            "pcs_at_Tref_ppm": float(pc),
            "ref_T_K": float(Tref),
        }

    globals_out = {"S1": S1, "S2": S2, "D2": D2, "D3": D3}
    return globals_out, Fi_map, diag

def make_plots(ds: Dataset, globals_out: Dict[str, float], Fi_map: Dict[str, float], outdir: Path, invert_x=True):
    outdir.mkdir(parents=True, exist_ok=True)
    S1, S2, D2, D3 = globals_out["S1"], globals_out["S2"], globals_out["D2"], globals_out["D3"]

    for key in sorted(ds.groups.keys(), key=natural_key):
        sub = ds.groups[key]
        T = sub["T"].to_numpy(float)
        d = sub["delta"].to_numpy(float)
        G = sub["G"].to_numpy(float)
        Fi = Fi_map[key]

        model = Fi * (S1/T + S2/(T**2)) + G * (D2/(T**2) + D3/(T**3))

        # 1) data vs model
        plt.figure()
        plt.scatter(T, d, s=20)
        plt.plot(T, model)
        if invert_x:
            plt.gca().invert_xaxis()
        plt.xlabel("Temperature (K)")
        plt.ylabel("delta_para (ppm)")
        plt.title(f"{key}: data vs model")
        plt.tight_layout()
        plt.savefig(outdir / f"{key}_fit.png", dpi=150)
        plt.close()

        # 2) residuals
        resid = d - model
        plt.figure()
        plt.scatter(T, resid, s=20)
        plt.axhline(0, linewidth=1)
        if invert_x:
            plt.gca().invert_xaxis()
        plt.xlabel("Temperature (K)")
        plt.ylabel("residual (ppm)")
        plt.title(f"{key}: residuals")
        plt.tight_layout()
        plt.savefig(outdir / f"{key}_residuals.png", dpi=150)
        plt.close()

def overall_metrics(ds: Dataset, globals_out: Dict[str, float], Fi_map: Dict[str, float]) -> Dict[str, float]:
    """Compute overall weighted SSE and RMSE across all nuclei/points."""
    S1, S2, D2, D3 = globals_out["S1"], globals_out["S2"], globals_out["D2"], globals_out["D3"]
    sse = 0.0
    wsum = 0.0
    npts = 0
    for key in sorted(ds.groups.keys(), key=natural_key):
        sub = ds.groups[key]
        T = sub["T"].to_numpy(float)
        d = sub["delta"].to_numpy(float)
        G = sub["G"].to_numpy(float)
        w = sub["w"].to_numpy(float)
        Fi = Fi_map[key]
        model = Fi * (S1/T + S2/(T**2)) + G * (D2/(T**2) + D3/(T**3))
        resid = d - model
        sse += float(np.sum(w * resid**2))
        wsum += float(np.sum(w))
        npts += len(d)
    rmse = math.sqrt(sse / wsum) if wsum > 0 else float("nan")
    return {"SSE": sse, "RMSE": rmse, "N_points": npts}

def compute_tau_metrics(globals_out: Dict[str, float], T_ref: float) -> Dict[str, float]:
    S1 = float(globals_out.get("S1", float("nan")))
    S2 = float(globals_out.get("S2", float("nan")))
    D2 = float(globals_out.get("D2", float("nan")))
    D3 = float(globals_out.get("D3", float("nan")))
    T  = float(T_ref)

    def safe_div(a, b):
        if not np.isfinite(a) or not np.isfinite(b) or b == 0.0:
            return float("nan")
        return a / b

    tau1 = safe_div(S2, S1)   # τ1 = S2/S1
    tau2 = safe_div(D3, D2)   # τ2 = D3/D2

    def frac(tau):
        x = safe_div(abs(tau), T)          # |τ|/T_ref
        if not np.isfinite(x):
            return float("nan")
        return x / (1.0 + x)               # x/(1+x)

    return {
        "tau_1": tau1,
        "tau_2": tau2,
        "tau_1_over_Tref": safe_div(tau1, T),
        "tau_1_frac": frac(tau1),
        "tau_2_over_Tref": safe_div(tau2, T),
        "tau_2_frac": frac(tau2),
        "T_ref_used_K": float(T),
    }

# Δχ_ax(T) calculation (ppm -> fraction)
def delta_chi_ax_series(T_array, D2, D3=0.0):
    T = np.asarray(T_array, dtype=float)
    X_ppm = D2 / (T**2) + D3 / (T**3)
    X_frac = X_ppm * 1e-6
    return 12 * np.pi * X_frac

def zero_intercept_ols(x: np.ndarray, y: np.ndarray):
    """절편=0 OLS: y ≈ m*x. 반환 (m, se_m, dof)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size < 2:
        return float("nan"), float("nan"), 0
    Sxx = float(np.sum(x*x))
    Sxy = float(np.sum(x*y))
    m = Sxy / Sxx
    r = y - m*x
    dof = max(x.size - 1, 1)
    s2 = float(np.sum(r*r)) / dof
    se_m = math.sqrt(s2 / Sxx)
    return m, se_m, dof

def zero_intercept_wls(x: np.ndarray, y: np.ndarray, sigma_y: np.ndarray):
    """절편=0 가중 OLS: y ≈ m*x, 가중치 w=1/σ_y^2. 반환 (m, se_m, dof)."""
    x = np.asarray(x, float); y = np.asarray(y, float); sy = np.asarray(sigma_y, float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(sy) & (sy > 0)
    x = x[mask]; y = y[mask]; sy = sy[mask]
    if x.size < 2:
        return float("nan"), float("nan"), 0
    w = 1.0 / (sy*sy)
    Sxx = float(np.sum(w * x*x))
    Sxy = float(np.sum(w * x*y))
    m = Sxy / Sxx
    r = y - m*x
    dof = max(x.size - 1, 1)
    s2 = float(np.sum(w * r*r)) / dof
    se_m = math.sqrt(s2 / Sxx)
    return m, se_m, dof

# -----------------------------
# Qt table models for results
# -----------------------------
class DictTableModel(QAbstractTableModel):
    def __init__(self, data: Dict[str, float], parent=None):
        super().__init__(parent)
        self.keys = list(data.keys())
        self.values = [data[k] for k in self.keys]

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self.keys)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 2

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        r, c = index.row(), index.column()
        if c == 0:
            return self.keys[r]
        if c == 1:
            v = self.values[r]
            if isinstance(v, float):
                return f"{v:.6g}"
            return str(v)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return ["Parameter", "Value"][section]
        return str(section)

class LinearTableModel(QAbstractTableModel):
    def __init__(self, results: Dict[str, dict], parent=None):
        super().__init__(parent)
        self.headers = [
            "Nucleus", "a (=Fi*S1)", "b (=Gi*D2)",
            "fcs_at_Tref (ppm)", "pcs_at_Tref (ppm)",
            "D2_i (=b/Gi)", "Δχ_ax(T_ref) per mol"
        ]
        self.rows = []
        for n in sorted(results.keys(), key=natural_key):
            v = results[n]
            self.rows.append((
                n, v["a1"], v["b1"],
                v["contact_Tref_ppm"], v["pcs_Tref_ppm"],
                v["D2_i"], v["dchi_Tref_m3_per_mol"]
            ))

    def rowCount(self, parent=QModelIndex()): return len(self.rows)
    def columnCount(self, parent=QModelIndex()): return len(self.headers)
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole: return None
        r = self.rows[index.row()][index.column()]
        if isinstance(r, float): return f"{r:.6g}"
        return r
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole: return None
        if orientation == Qt.Orientation.Horizontal: return self.headers[section]
        return str(section+1)

def two_term_wls(x1: np.ndarray, x2: np.ndarray, y: np.ndarray, w: np.ndarray):
    """
    절편 없는 WLS: y ≈ a*x1 + b*x2
    반환: a, b, sa, sb  (추정치와 표준오차)
    정책:
      - n < 2: a,b 불가 → 전부 NaN
      - n = 2: a,b 가능 → sa,sb는 NaN
      - n >= 3: sa,sb 계산
    """
    x1 = np.asarray(x1, float); x2 = np.asarray(x2, float)
    y  = np.asarray(y,  float); w  = np.asarray(w,  float)

    # finite + w>0만 사용
    mask = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x1 = x1[mask]; x2 = x2[mask]; y = y[mask]; w = w[mask]

    n = int(len(y))
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    X = np.column_stack([x1, x2])
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    XtWX = Xw.T @ Xw
    XtWy = Xw.T @ yw

    try:
        beta = np.linalg.solve(XtWX, XtWy)
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        XtWX_inv = np.linalg.pinv(XtWX)
        beta = XtWX_inv @ XtWy

    a, b = float(beta[0]), float(beta[1])

    # n=2이면 SE는 정의 불가 → NaN
    if n < 3:
        return a, b, float("nan"), float("nan")

    r = y - X @ beta
    dof = n - 2  # >=1
    sigma2 = float((sw * r) @ (sw * r)) / dof
    cov = sigma2 * XtWX_inv
    sa = math.sqrt(float(cov[0,0]))
    sb = math.sqrt(float(cov[1,1]))
    return a, b, sa, sb

class FiTableModel(QAbstractTableModel):
    """Display nucleus-wise Fi."""
    def __init__(self, fi_map: Dict[str, float], parent=None):
        super().__init__(parent)
        self.rows = sorted(fi_map.items(), key=lambda kv: natural_key(kv[0]))  # (nucleus, Fi)

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self.rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 2

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        r, c = index.row(), index.column()
        k, v = self.rows[r]
        if c == 0:
            return k
        if c == 1:
            return f"{v:.6g}"
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return ["Nucleus", "Fi"][section]
        return str(section)

class DiagTableModel(QAbstractTableModel):
    """Display diagnostics per nucleus (dict of dicts)."""
    def __init__(self, diag: Dict[str, dict], fi_map: Optional[Dict[str, float]] = None, parent=None):
        super().__init__(parent)
        self.keys = sorted(diag.keys(), key=natural_key)
        self.data_dict = diag
        self.fi_map = fi_map or {}
        self.cols = ["n_points", "T_min_K", "T_max_K", "RMSE_ppm",
                     "Fi", "fcs_at_Tref_ppm", "pcs_at_Tref_ppm", "ref_T_K"]

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self.keys)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 1 + len(self.cols)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        r, c = index.row(), index.column()
        key = self.keys[r]
        if c == 0:
            return key
        colname = self.cols[c - 1]
        if colname == "Fi":
            v = self.fi_map.get(key, "")
        else:
            v = self.data_dict.get(key, {}).get(colname, "")
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            headers = ["Nucleus"] + self.cols
            return headers[section] if 0 <= section < len(headers) else ""
        return str(section + 1)

class DchiPlotWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Δχ_ax vs Temperature")
        self.resize(640, 480)

        layout = QVBoxLayout(self)

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

    def plot(self, T_grid, dchi_ext, dchi_base=None):
        self.ax.clear()
        self.ax.plot(T_grid, dchi_ext, label="Extended", color="blue")
        if dchi_base is not None:
            self.ax.plot(T_grid, dchi_base, label="Baseline", color="red", linestyle="--")
        self.ax.set_xlabel("Temperature (K)")
        self.ax.set_ylabel("Δχ_ax (dimensionless, per molecule)")
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

class CopyableTableView(QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)
        # multi drag
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        # context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)

    def _show_menu(self, pos):
        menu = QMenu(self)
        act_copy = QAction("Copy", self)
        act_copy.setShortcut(QKeySequence.StandardKey.Copy)
        act_copy.triggered.connect(self.copy_selection)
        menu.addAction(act_copy)
        menu.exec(self.viewport().mapToGlobal(pos))

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy_selection()
        else:
            super().keyPressEvent(event)

    def copy_selection(self):
        indexes = self.selectedIndexes()
        if not indexes:
            return
        rows = sorted(set(i.row() for i in indexes))
        cols = sorted(set(i.column() for i in indexes))
        grid = []
        for r in rows:
            row_items = []
            for c in cols:
                idx = self.model().index(r, c)
                val = self.model().data(idx, Qt.ItemDataRole.DisplayRole)
                row_items.append("" if val is None else str(val))
            grid.append("\t".join(row_items))
        text = "\n".join(grid)
        QGuiApplication.clipboard().setText(text)

# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ParaFit")
        self.resize(950, 830)

        self.df: Optional[pd.DataFrame] = None
        self.dataset: Optional[Dataset] = None
        self.globals_out: Optional[Dict[str, float]] = None
        self.fi_map: Optional[Dict[str, float]] = None
        self.diag: Optional[Dict[str, dict]] = None
        self.csv_path: Optional[Path] = None

        # ORCA
        self.orca_out_path: Optional[Path] = None
        self.orca_chi_series = None  # ChiTensorSeries
        self.orca_ailft = None       # AILFTResult

        # baseline results
        self.baseline_globals: Optional[Dict[str, float]] = None
        self.baseline_Fi: Optional[Dict[str, float]] = None
        self.baseline_diag: Optional[Dict[str, dict]] = None
        self.baseline_metrics: Optional[Dict[str, float]] = None

        central = QWidget(self)
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        # --- file and column mapping ---
        file_row = QHBoxLayout()
        self.lb_file = QLabel("No CSV loaded")
        self.lb_file.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.lb_file.setMinimumWidth(0)
        self.lb_file.setWordWrap(False)
        btn_save_sample = QPushButton("Save Sample CSV")
        btn_save_sample.clicked.connect(self.on_save_sample_csv)
        btn_load = QPushButton("Load CSV")
        btn_load.clicked.connect(self.on_load_csv)
        btn_load_orca_chi = QPushButton("Load ORCA .out (χ mode)")
        btn_load_orca_chi.clicked.connect(self.on_load_orca_chi)
        btn_load_orca_ailft = QPushButton("Load ORCA .out (AILFT mode)")
        btn_load_orca_ailft.clicked.connect(self.on_load_orca_ailft)
        file_row.addWidget(self.lb_file, 1)
        file_row.addWidget(btn_save_sample, 0)
        file_row.addWidget(btn_load, 0)
        file_row.addWidget(btn_load_orca_chi, 0)
        file_row.addWidget(btn_load_orca_ailft, 0)
        layout.addLayout(file_row)

        btn_view_orca = QPushButton("View ORCA table")
        btn_view_orca.clicked.connect(self.on_view_orca_table)
        file_row.addWidget(btn_view_orca, 0)
        self.btn_view_orca = btn_view_orca
        self.btn_view_orca.setEnabled(False)
        self._orca_viewer = None

        map_box = QGroupBox("Column mapping")
        map_form = QFormLayout(map_box)
        self.cmb_nuc = QComboBox();
        self.cmb_T = QComboBox()
        self.cmb_delta = QComboBox();
        self.cmb_G = QComboBox()
        self.cmb_w = QComboBox();
        self.cmb_pcs_guess = QComboBox()
        self.cmb_w.setEditable(False)
        map_form.addRow("nucleus", self.cmb_nuc)
        map_form.addRow("T (K)", self.cmb_T)
        map_form.addRow("delta_para (ppm)", self.cmb_delta)
        map_form.addRow("G_i", self.cmb_G)
        map_form.addRow("weight (optional)", self.cmb_w)
        map_form.addRow("PCS_guess (ppm, optional)", self.cmb_pcs_guess)

        # --- options ---
        opt_box = QGroupBox("Fitting options")
        opt_form = QFormLayout(opt_box)

        # init edits
        self.ed_init_S1 = QLineEdit("100.0")
        self.ed_init_S2 = QLineEdit("0.0")
        self.ed_init_D2 = QLineEdit("1e4")
        self.ed_init_D3 = QLineEdit("0.0")

        # fix checks + edits
        self.chk_fix_S1 = QCheckBox("Fix S1")
        self.chk_fix_S2 = QCheckBox("Fix S2")
        self.chk_fix_D2 = QCheckBox("Fix D2")
        self.chk_fix_D3 = QCheckBox("Fix D3")

        self.ed_fix_S1 = QLineEdit();
        self.ed_fix_S1.setPlaceholderText("value if fixed")
        self.ed_fix_S2 = QLineEdit();
        self.ed_fix_S2.setPlaceholderText("value if fixed")
        self.ed_fix_D2 = QLineEdit();
        self.ed_fix_D2.setPlaceholderText("value if fixed")
        self.ed_fix_D3 = QLineEdit();
        self.ed_fix_D3.setPlaceholderText("value if fixed")

        for w in (self.ed_init_S1, self.ed_init_S2, self.ed_init_D2, self.ed_init_D3,
                  self.ed_fix_S1, self.ed_fix_S2, self.ed_fix_D2, self.ed_fix_D3):
            w.setMaximumWidth(120)
            w.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # ---- (Init + Fix) grid ----
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        rows = [
            ("Init S1", self.ed_init_S1, self.chk_fix_S1, self.ed_fix_S1),
            ("Init S2", self.ed_init_S2, self.chk_fix_S2, self.ed_fix_S2),
            ("Init D2", self.ed_init_D2, self.chk_fix_D2, self.ed_fix_D2),
            ("Init D3", self.ed_init_D3, self.chk_fix_D3, self.ed_fix_D3),
        ]

        for r, (lab, init_ed, chk, fix_ed) in enumerate(rows):
            grid.addWidget(QLabel(lab), r, 0)
            grid.addWidget(init_ed, r, 1)
            grid.addWidget(chk, r, 2)
            grid.addWidget(fix_ed, r, 3)

        opt_form.addRow(grid)

        self.ed_ridge = QLineEdit("0.0")
        opt_form.addRow("Ridge λ on Fi", self.ed_ridge)

        self.ed_pcs_prior_lambda = QLineEdit("0.0")
        opt_form.addRow("PCS guess prior λ (1/ppm²)", self.ed_pcs_prior_lambda)

        self.ed_Tref_linear = QLineEdit("298.0")
        opt_form.addRow("T_ref (K) for analysis", self.ed_Tref_linear)

        self.chk_run_linear = QCheckBox("Run linear regression δ = a/T + b/T²")
        self.chk_run_linear.setChecked(True)
        opt_form.addRow(self.chk_run_linear)

        self.chk_run_baseline = QCheckBox("Run baseline (classical S2=0, D3=0) first")
        self.chk_run_baseline.setChecked(True)
        opt_form.addRow(self.chk_run_baseline)

        self.chk_invertx = QCheckBox("Invert X-axis for plots (VT style)")
        self.chk_invertx.setChecked(False)
        opt_form.addRow(self.chk_invertx)

        group_row = QHBoxLayout()
        group_row.addWidget(map_box, 1)  # Column mapping
        group_row.addWidget(opt_box, 1)  # Fitting options
        layout.addLayout(group_row)

        self.dchi_window = None  # Δχ_ax plot window

        # --- buttons ---
        btn_row = QHBoxLayout()
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.clicked.connect(self.on_fit)
        self.btn_sweep_ridge = QPushButton("Sweep Ridge λ")
        self.btn_sweep_ridge.clicked.connect(self.on_sweep_ridge)
        self.btn_save_json = QPushButton("Save JSON")
        self.btn_save_json.clicked.connect(self.on_save_json)
        self.btn_save_plots = QPushButton("Save Plots")
        self.btn_save_plots.clicked.connect(self.on_save_plots)
        self.btn_save_data = QPushButton("Save Data (CSV)")
        self.btn_save_data.clicked.connect(self.on_save_data)
        self.btn_save_dchi = QPushButton("Save Δχ_ax vs T")
        self.btn_save_dchi.clicked.connect(self.on_save_dchi)
        self.btn_view_dchi = QPushButton("View Δχ_ax Plot")
        self.btn_view_dchi.clicked.connect(self.on_view_dchi)

        for b in (self.btn_fit, self.btn_sweep_ridge, self.btn_save_json, self.btn_save_plots,
                  self.btn_save_data, self.btn_save_dchi, self.btn_view_dchi):
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        # --- results area ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_panel = QWidget(); left_layout = QVBoxLayout(left_panel)
        right_panel = QWidget(); right_layout = QVBoxLayout(right_panel)

        self.tbl_linear = CopyableTableView()
        left_layout.addWidget(QLabel("Linear approximation (δ = a/T + b/T²)"))
        left_layout.addWidget(self.tbl_linear)

        self.tbl_globals = CopyableTableView()
        self.tbl_diag = CopyableTableView()
        left_layout.addWidget(QLabel("Global parameters (S1,S2,D2,D3) — Extended model"))
        left_layout.addWidget(self.tbl_globals)

        self.txt_log = QTextEdit(); self.txt_log.setReadOnly(True)
        right_layout.addWidget(QLabel("Diagnostics per nucleus — Extended model"))
        right_layout.addWidget(self.tbl_diag)
        right_layout.addWidget(QLabel("Log / Messages (baseline & extended summaries)"))
        right_layout.addWidget(self.txt_log)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([620, 620])
        layout.addWidget(splitter)

        self.set_controls_enabled(False)

    def on_sweep_ridge(self):
        """
        현재 CSV/설정 기준으로 λ 리스트를 받아 Extended 모델을 λ별로 반복 적합.
        결과를 새 창(RidgeSweepWindow)에서 플롯/표로 표시.
        """
        if self.df is None or self.dataset is None:
            QMessageBox.warning(self, "No data", "Please load a CSV and run at least once to set columns.")
            return

        # λ input
        text, ok = QInputDialog.getText(
            self, "Ridge λ sweep",
            "Enter 'lambda_min, lambda_max, N_points, include_zero(0/1)':",
            text="1e-10, 1e2, 50, 1"
        )
        if not ok or not text.strip():
            return

        try:
            parts = [p.strip() for p in text.split(",")]
            if len(parts) < 3:
                raise ValueError("Need at least 3 values.")
            lam_min = float(eval(parts[0], {}, {}))
            lam_max = float(eval(parts[1], {}, {}))
            n_pts = int(eval(parts[2], {}, {}))
            include0 = True
            if len(parts) >= 4:
                include0 = bool(int(eval(parts[3], {}, {})))

            if not (np.isfinite(lam_min) and np.isfinite(lam_max)):
                raise ValueError
            if lam_min <= 0 or lam_max <= 0 or lam_max <= lam_min:
                raise ValueError("lambda_min/lambda_max must be > 0 and max>min.")
            if n_pts < 2:
                raise ValueError("N_points must be >= 2.")

            lam_list = np.logspace(np.log10(lam_min), np.log10(lam_max), n_pts).tolist()
            if include0:
                lam_list = [0.0] + lam_list

        except Exception as e:
            QMessageBox.critical(self, "Invalid input", f"Bad sweep settings:\n{e}")
            return

        # UI에서 현재 init/fix/Tref 읽기 (Fit과 동일 로직 재사용)
        try:
            init = {
                "S1": float(eval(self.ed_init_S1.text(), {}, {})),
                "S2": float(eval(self.ed_init_S2.text(), {}, {})),
                "D2": float(eval(self.ed_init_D2.text(), {}, {})),
                "D3": float(eval(self.ed_init_D3.text(), {}, {})),
            }
        except Exception:
            QMessageBox.critical(self, "Error", "Invalid initial guesses (S1,S2,D2,D3).")
            return

        fix_ext: Dict[str, float] = {}
        try:
            if self.chk_fix_S1.isChecked(): fix_ext["S1"] = float(eval(self.ed_fix_S1.text(), {}, {}))
            if self.chk_fix_S2.isChecked(): fix_ext["S2"] = float(eval(self.ed_fix_S2.text(), {}, {}))
            if self.chk_fix_D2.isChecked(): fix_ext["D2"] = float(eval(self.ed_fix_D2.text(), {}, {}))
            if self.chk_fix_D3.isChecked(): fix_ext["D3"] = float(eval(self.ed_fix_D3.text(), {}, {}))
        except Exception:
            QMessageBox.critical(self, "Error", "Invalid fixed parameter value(s).")
            return

        try:
            Tref = float(self.ed_Tref_linear.text())
        except Exception:
            Tref = 298.0

        # 스윕 실행
        try:
            df_sweep = sweep_ridge_lambda(self.dataset, init, fix_ext, lam_list, Tref)
        except Exception as e:
            QMessageBox.critical(self, "Sweep failed", f"Ridge sweep failed:\n{e}")
            return

        # 창 생성/표시
        if not hasattr(self, "_ridge_win") or self._ridge_win is None:
            self._ridge_win = RidgeSweepWindow(self)
            self._ridge_win.setWindowFlags(Qt.WindowType.Window)
        self._ridge_win.set_results(df_sweep)
        self._ridge_win.show()
        self._ridge_win.raise_()

        # 로그에 요약
        self.append_log("[Ridge sweep] λ list: " + ", ".join(str(l) for l in lam_list))
        if "error" in df_sweep.columns and df_sweep["error"].notna().any():
            n_err = int(df_sweep["error"].notna().sum())
            self.append_log(f"[Ridge sweep] {n_err} cases failed (see sweep window table).")
        if not df_sweep.empty and "RMSE" in df_sweep.columns:
            best = df_sweep.loc[df_sweep["RMSE"].astype(float).idxmin()]
            self.append_log(f"[Ridge sweep] Best RMSE at λ={best['lambda']}: {best['RMSE']:.6g} ppm")

    def set_controls_enabled(self, enabled: bool):
        for w in [self.cmb_nuc, self.cmb_T, self.cmb_delta, self.cmb_G, self.cmb_w, self.cmb_pcs_guess,
                  self.ed_init_S1, self.ed_init_S2, self.ed_init_D2, self.ed_init_D3,
                  self.chk_fix_S1, self.chk_fix_S2, self.chk_fix_D2, self.chk_fix_D3, self.ed_pcs_prior_lambda,
                  self.ed_fix_S1, self.ed_fix_S2, self.ed_fix_D2, self.ed_fix_D3,
                  self.ed_ridge, self.chk_invertx, self.chk_run_baseline, self.chk_run_linear,
                  self.ed_Tref_linear, self.btn_fit, self.btn_save_json, self.btn_save_plots, self.btn_save_data,
                  self.btn_save_dchi, self.btn_view_dchi, self.btn_sweep_ridge]:
            w.setEnabled(enabled)

    # ------------- Actions -------------
    def on_load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV files (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read CSV:\n{e}")
            return

        self.df = df
        self.csv_path = Path(path)
        self.lb_file.setText(self.csv_path.name)  # file name
        self.lb_file.setToolTip(str(self.csv_path))  # full path - mouse hover

        # populate combo boxes with columns
        cols = list(df.columns)
        for cmb in [self.cmb_nuc, self.cmb_T, self.cmb_delta, self.cmb_G, self.cmb_w, self.cmb_pcs_guess]:
            cmb.clear()
            cmb.addItems(cols)
            cmb.insertItem(0, "<none>")
            cmb.setCurrentIndex(0)
        # sensible defaults if matching names exist
        def set_if_found(cmb: QComboBox, name: str):
            idx = cmb.findText(name)
            if idx >= 0: cmb.setCurrentIndex(idx)

        set_if_found(self.cmb_nuc, "nucleus")
        set_if_found(self.cmb_T, "T_K")
        set_if_found(self.cmb_delta, "delta_para_ppm")
        set_if_found(self.cmb_G, "G_i")
        set_if_found(self.cmb_w, "weight")
        set_if_found(self.cmb_pcs_guess, "PCS_guess_ppm")

        self.set_controls_enabled(True)
        self.append_log(f"Loaded CSV with {len(df)} rows and columns: {cols}")

    def on_view_orca_table(self):
        if self.orca_chi_series is None:
            QMessageBox.warning(self, "No ORCA data", "Load ORCA χ(T) output first.")
            return

        if self._orca_viewer is None:
            self._orca_viewer = OrcaChiViewer(None)
            self._orca_viewer.setWindowFlags(Qt.WindowType.Window)

        # try to pass CSV mapping too (for Tab2)
        df = self.df
        nuc_col = self.cmb_nuc.currentText()
        if nuc_col in (None, "", "<none>"):
            nuc_col = None
        T_col = self.cmb_T.currentText() if df is not None else None
        if T_col in (None, "", "<none>"):
            T_col = None
        G_col = self.cmb_G.currentText() if df is not None else None
        if G_col in (None, "", "<none>"):
            G_col = None
        self._orca_viewer.set_inputs(OrcaViewerInputs(
            series=self.orca_chi_series,
            df=df,
            nuc_col=nuc_col,
            T_col=T_col,
            G_col=G_col,
        ))
        self._orca_viewer.show()
        self._orca_viewer.raise_()

    def on_load_orca_chi(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open ORCA output", "", "ORCA out (*.out *.log *.txt);;All files (*.*)"
        )
        if not path:
            return

        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
            series = read_orca_temp_dependent_chiT_tensor(text, use_4pi=True)  # per-molecule m^3
        except Exception as e:
            QMessageBox.critical(self, "ORCA parse error", f"Failed to parse ORCA χ tensor blocks:\n{e}")
            return

        self.orca_out_path = Path(path)
        self.orca_chi_series = series
        self.append_log(
            f"[ORCA] Loaded χ(T) tensors from: {self.orca_out_path.name} "
            f"(n={len(series.tensors_per_molecule_m3)})"
        )

        # Tref
        try:
            Tref = float(eval(self.ed_Tref_linear.text(), {}, {}))
        except Exception:
            Tref = 298.0

        # compute Δχ at Tref for logging (per molecule, m^3)
        try:
            chi = interpolate_tensor(series, Tref)
            chi = make_traceless(chi)
            vals, _ = mehring_order_from_tensor(chi)
            chi_xx, chi_yy, chi_zz = map(float, vals)
            d_ax, d_rh = delta_chi_ax_rh_from_principal(chi_xx, chi_yy, chi_zz)
            self.append_log(f"[ORCA] Δχ_ax(Tref={Tref:.2f} K) (m^3/molecule, traceless+Mehring): {d_ax:.6e}")
            self.append_log(f"[ORCA] Δχ_rh(Tref={Tref:.2f} K) (m^3/molecule, traceless+Mehring): {d_rh:.6e}")
        except Exception as e:
            self.append_log(f"[ORCA] Warning: could not compute Δχ at Tref: {e}")

        # attach PCS_guess_orca_ppm if CSV already loaded and has T & Gi_raw selected
        if self.df is None:
            return

        col_T = self.cmb_T.currentText()
        col_Graw = self.cmb_G.currentText()  # this is Gi_raw (Å^-3) in your convention

        if col_T == "<none>" or col_Graw == "<none>":
            self.append_log("[ORCA] CSV column mapping incomplete (T/G not selected). Skipping PCS_guess column.")
            return
        if col_T not in self.df.columns or col_Graw not in self.df.columns:
            self.append_log(f"[ORCA] CSV mapping invalid: T='{col_T}', G='{col_Graw}'. Skipping PCS_guess column.")
            return

        try:
            df2 = self.df.copy()

            # ---- range check (to warn about clamping) ----
            Ts_avail = sorted(series.tensors_per_molecule_m3.keys())
            Tmin, Tmax = float(Ts_avail[0]), float(Ts_avail[-1])

            # count invalid and out-of-range for log
            Tvals = pd.to_numeric(df2[col_T], errors="coerce").to_numpy(float)
            Gvals = pd.to_numeric(df2[col_Graw], errors="coerce").to_numpy(float)
            n_bad = int(np.sum(~np.isfinite(Tvals) | ~np.isfinite(Gvals)))
            n_oob = int(np.sum(np.isfinite(Tvals) & ((Tvals < Tmin) | (Tvals > Tmax))))

            # ---- compute PCS guess in ppm using your convention ----
            tag = Path(path).stem[:24]
            new_col = f"PCS_guess_orca_ppm__{tag}"

            # uses: Gi_raw (Å^-3) and Δχ(m^3/molecule) -> PCS(ppm) = Gi_raw * Δχ * 1e36 / (12π)
            df2 = attach_pcs_from_chi(
                df2,
                series,
                T_col=col_T,
                Gax_raw_col=col_Graw,
                Grh_raw_col=None,  # set if you later have a Grh column
                out_col=new_col,
                traceless=True,
                mehring=True,
                include_12pi=True,
            )

            self.df = df2

            # refresh combobox columns so new col appears
            cols = list(self.df.columns)
            for cmb in [self.cmb_nuc, self.cmb_T, self.cmb_delta, self.cmb_G, self.cmb_w, self.cmb_pcs_guess]:
                cur = cmb.currentText()
                cmb.blockSignals(True)
                cmb.clear()
                cmb.insertItem(0, "<none>")
                cmb.addItems(cols)
                idx2 = cmb.findText(cur)
                cmb.setCurrentIndex(idx2 if idx2 >= 0 else 0)
                cmb.blockSignals(False)

            # auto-select pcs_guess col
            idx3 = self.cmb_pcs_guess.findText(new_col)
            if idx3 >= 0:
                self.cmb_pcs_guess.setCurrentIndex(idx3)

            self.append_log(f"[ORCA] Added column '{new_col}' (PCS guess in ppm; Gi_raw=Å^-3, χ per-molecule m^3).")
            self.append_log(
                f"[ORCA] ORCA χ(T) grid: Tmin={Tmin:.2f} K, Tmax={Tmax:.2f} K. "
                f"Rows with invalid T/G: {n_bad}. Rows outside grid (clamped): {n_oob}."
            )

        except Exception as e:
            self.append_log(f"[ORCA] Could not attach PCS_guess column: {e}")

        try:
            D2, D3, rmse = estimate_D2_D3_from_orca_chi_ax(self.orca_chi_series, traceless=True, mehring=True)
            self.append_log("[ORCA-fit] Fit Δχ_ax(T) ≈ (D2/1e6)/T^2 + (D3/1e6)/T^3  (Δχ in m^3/molecule)")
            self.append_log(f"[ORCA-fit] Suggested init for Extended model (when G is SI-scaled): "
                            f"D2={D2:.6e}, D3={D3:.6e}  (units: ppm·m^3·K^n/molecule), "
                            f"RMSE(Δχ)={rmse:.3e} m^3/molecule")
            if hasattr(self, "ed_D2_init"):
                self.ed_init_D2.setText(f"{D2:.6e}")
            if hasattr(self, "ed_D3_init"):
                self.ed_init_D3.setText(f"{D3:.6e}")

        except Exception as e:
            self.append_log(f"[ORCA-fit] Could not estimate D2/D3 from ORCA: {e}")

        self.btn_view_orca.setEnabled(True)

    def on_load_orca_ailft(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open ORCA output", "", "ORCA out (*.out *.log *.txt);;All files (*.*)")
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
            ailft = parse_orca_ailft(text)
        except Exception as e:
            QMessageBox.critical(self, "ORCA parse error", f"Failed to parse ORCA AILFT block:\n{e}")
            return

        self.orca_out_path = Path(path)
        self.orca_ailft = ailft

        # log summary
        self.append_log(f"[ORCA] Loaded AILFT from: {self.orca_out_path.name}")
        self.append_log(f"  config={ailft.configuration}, shell={ailft.shell_type}, CI_blocks={ailft.n_ci_blocks}, MOs={ailft.active_mo_range}, center_atom={ailft.metal_center_atom}")
        if ailft.soc_constant_cm1 is not None:
            self.append_log(f"  SOC zeta (cm^-1): {ailft.soc_constant_cm1:.2f}")
        if ailft.soc_constants:
            tags = ", ".join([f"{k}={v:.2f}" for k, v in ailft.soc_constants.items()])
            self.append_log(f"  SOC tags: {tags}")
        if ailft.slater_condon_cm1:
            tags = ", ".join([f"{k}={v:.1f}" for k, v in ailft.slater_condon_cm1.items()])
            self.append_log(f"  Slater-Condon (cm^-1): {tags}")
        if ailft.racah_cm1:
            tags = ", ".join([f"{k}={v:.1f}" for k, v in ailft.racah_cm1.items()])
            self.append_log(f"  Racah (cm^-1): {tags}")
        if ailft.lf_eigenfunctions:
            self.append_log(f"  LF eigenfunctions parsed: n={len(ailft.lf_eigenfunctions)} (show first 3)")
            for lev in ailft.lf_eigenfunctions[:3]:
                self.append_log(f"    {lev.idx}: {lev.energy_cm1:.1f} cm^-1 ({lev.energy_ev:.3f} eV)")
        if ailft.vlft_au is not None:
            try:
                evals = np.linalg.eigvalsh(ailft.vlft_au)
                evals_cm1 = evals * 219474.6313705
                self.append_log(f"  VLFT: dim={ailft.vlft_au.shape[0]} | one-electron levels (cm^-1, rel): "
                                f"{', '.join(f'{(x - evals_cm1.min()):.1f}' for x in evals_cm1)}")
            except Exception:
                self.append_log("  VLFT: parsed (eigenvalue summary failed)")

    def on_save_sample_csv(self):
        """Create and save a template CSV with expected headers and synthetic rows."""
        out_path, _ = QFileDialog.getSaveFileName(self, "Save Sample CSV",
                                                  "example_vt_data.csv",
                                                  "CSV (*.csv)")
        if not out_path:
            return
        # Generate minimal synthetic data
        nuclei = ["H3", "H5", "H6", "C1", "C3"]
        Tvals = np.array([278, 288, 298, 308, 318, 328, 338], dtype=float)
        Gi_map = {"H3": 2.1, "H5": 1.3, "H6": 1.0, "C1": 2.5, "C3": 1.8}
        S1_true, S2_true, D2_true, D3_true = 150.0, 30.0, 9.0e3, -2.0e3
        Fi_true = {"H3": 0.05, "H5": 0.09, "H6": 0.04, "C1": -1.2, "C3": -2.0}

        rows = []
        rng = np.random.default_rng(0)
        for nuc in nuclei:
            Gi = Gi_map[nuc]
            for T in Tvals:
                delta = Fi_true[nuc] * (S1_true / T + S2_true / T ** 2) + Gi * (D2_true / T ** 2 + D3_true / T ** 3)
                delta += rng.normal(0.02)
                rows.append({
                    "nucleus": nuc,
                    "T_K": float(T),
                    "delta_para_ppm": float(delta),
                    "G_i": float(Gi),
                    "weight": 1.0,
                    "PCS_guess_ppm": np.nan  # ← 새 열
                })

        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        self.append_log(f"Saved sample CSV: {out_path}")
        self.append_log("Sample CSV includes a 'PCS_guess_ppm' column (empty by default).")

    def update_dchi_plot(self, T_min=200, T_max=400, step=1.0):
        if not self.globals_out:
            return
        import numpy as np
        T_grid = np.arange(T_min, T_max + step, step)

        # Extended
        D2_ext = float(self.globals_out["D2"])
        D3_ext = float(self.globals_out.get("D3", 0.0))
        dchi_ext = delta_chi_ax_series(T_grid, D2_ext, D3_ext)

        # Baseline
        dchi_base = None
        if self.baseline_globals:
            D2_base = float(self.baseline_globals["D2"])
            dchi_base = delta_chi_ax_series(T_grid, D2_base, 0.0)

        # 새 창 없으면 생성
        if self.dchi_window is None:
            self.dchi_window = DchiPlotWindow(None)
            self.dchi_window.setWindowFlags(Qt.WindowType.Window)

        self.dchi_window.plot(T_grid, dchi_ext, dchi_base)
        self.dchi_window.show()
        self.dchi_window.raise_()

    def on_view_dchi(self):
        if not self.globals_out:
            QMessageBox.warning(self, "No fit", "Please run a fit first.")
            return

        import numpy as np
        T_grid = np.arange(200, 401, 1.0)  # default 200–400 K, 1K step

        # Extended
        D2_ext = float(self.globals_out["D2"])
        D3_ext = float(self.globals_out.get("D3", 0.0))
        dchi_ext = delta_chi_ax_series(T_grid, D2_ext, D3_ext)

        # Baseline
        dchi_base = None
        if self.baseline_globals:
            D2_base = float(self.baseline_globals["D2"])
            dchi_base = delta_chi_ax_series(T_grid, D2_base, 0.0)

        if self.dchi_window is None:
            self.dchi_window = DchiPlotWindow(None)
            self.dchi_window.setWindowFlags(Qt.WindowType.Window)

        self.dchi_window.plot(T_grid, dchi_ext, dchi_base)
        self.dchi_window.show()
        self.dchi_window.raise_()

    def on_fit(self):
        if self.df is None:
            QMessageBox.warning(self, "No data", "Please load a CSV first.")
            return

        # ---------- read column selections (Gi optional) ----------
        try:
            col_nuc = self.get_cmb_value(self.cmb_nuc, required=True, name="nucleus")
            col_T = self.get_cmb_value(self.cmb_T, required=True, name="T (K)")
            col_del = self.get_cmb_value(self.cmb_delta, required=True, name="delta_para (ppm)")

            tmp_G = self.get_cmb_value(self.cmb_G, required=False)
            col_G = None if (tmp_G in ("", "<none>")) else tmp_G  # ← Gi를 옵션으로

            col_w = self.get_cmb_value(self.cmb_w, required=False)
            col_pcs_guess = self.get_cmb_value(self.cmb_pcs_guess, required=False)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        # ---------- dataset 준비 (Gi 유무를 내부 플래그로 보관) ----------
        try:
            ds = prepare_dataset(
                self.df, col_nuc, col_T, col_del,
                col_G if col_G else None,
                col_w if col_w not in ("", "<none>") else None,
                col_pcs_guess if col_pcs_guess not in ("", "<none>") else None
            )

            # Gi가 있을 때만 SI 스케일링(G_raw → G) 적용
            if getattr(ds, "has_G", False):
                # prepare_dataset에서 이미 G_raw를 만들어 두었다고 가정
                K = 1e30 / (12 * math.pi)  # ≈ 2.6525823848649222e28
                for key in sorted(ds.groups.keys(), key=natural_key):
                    sub = ds.groups[key]
                    ds.groups[key].loc[:, "G"] = sub["G_raw"].astype(float) * K
                self.append_log(f"G_i scaled to SI with K={K:g} (now in 1/(12π)·m^-3).")
            else:
                self.append_log(
                    "No Gi column provided → running in linear-only mode (no baseline/extended, no D2, no PCS vs Gi).")

            self.dataset = ds

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to prepare dataset:\n{e}")
            return

        # ---------- options ----------
        try:
            lambda_pcs_prior = float(eval(self.ed_pcs_prior_lambda.text(), {}, {}))
            lambda_pcs_prior = max(0.0, lambda_pcs_prior)
        except Exception:
            QMessageBox.critical(self, "Error", "Invalid PCS prior λ.")
            return

        try:
            init = {
                "S1": float(eval(self.ed_init_S1.text(), {}, {})),
                "S2": float(eval(self.ed_init_S2.text(), {}, {})),
                "D2": float(eval(self.ed_init_D2.text(), {}, {})),
                "D3": float(eval(self.ed_init_D3.text(), {}, {})),
            }
        except Exception:
            QMessageBox.critical(self, "Error", "Invalid initial guesses (S1,S2,D2,D3).")
            return

        try:
            Tref_ui = float(self.ed_Tref_linear.text())
        except Exception:
            Tref_ui = 298.0

        try:
            ridge = float(eval(self.ed_ridge.text(), {}, {}))
            self.dataset.ridge_lambda = max(0.0, ridge)
        except Exception:
            QMessageBox.critical(self, "Error", "Invalid ridge λ.")
            return

        fix_ext: Dict[str, float] = {}
        try:
            if self.chk_fix_S1.isChecked(): fix_ext["S1"] = float(eval(self.ed_fix_S1.text(), {}, {}))
            if self.chk_fix_S2.isChecked(): fix_ext["S2"] = float(eval(self.ed_fix_S2.text(), {}, {}))
            if self.chk_fix_D2.isChecked(): fix_ext["D2"] = float(eval(self.ed_fix_D2.text(), {}, {}))
            if self.chk_fix_D3.isChecked(): fix_ext["D3"] = float(eval(self.ed_fix_D3.text(), {}, {}))
        except Exception:
            QMessageBox.critical(self, "Error", "Invalid fixed parameter value(s).")
            return

        # ---------- Linear fit (δ ≈ a/T + b/T², no intercept) ----------
        if self.chk_run_linear.isChecked():
            try:
                T_ref = float(self.ed_Tref_linear.text())
            except Exception:
                T_ref = 298.0

            self.linear_results = {}
            for nuc in sorted(self.dataset.groups.keys(), key=natural_key):
                sub = self.dataset.groups[nuc]
                T = sub["T"].to_numpy(float)
                d = sub["delta"].to_numpy(float)
                w = sub["w"].to_numpy(float)

                # Gi는 있을 때만 평균값 사용, 없으면 NaN
                if "G_raw" in sub.columns:
                    vals = sub["G_raw"].to_numpy(float)
                    if vals.size == 0 or np.all(np.isnan(vals)):
                        Gi = float("nan")
                    else:
                        Gi = float(np.nanmean(vals))
                else:
                    Gi = float("nan")

                x1 = 1.0 / T
                x2 = 1.0 / (T ** 2)

                # Two-term WLS: δ ≈ a1*(1/T) + b1*(1/T^2)
                a1, b1, sa1, sb1 = two_term_wls(x1, x2, d, w)

                # T_ref FCS PCS separation(ppm)
                contact_Tref = a1 / T_ref
                pcs_Tref = b1 / (T_ref ** 2)

                # D2_i (= b1/Gi), Gi 없으면 NaN
                if np.isfinite(Gi) and Gi != 0.0 and np.isfinite(b1):
                    D2_i = b1 / Gi
                    D2_se_i = abs(sb1 / Gi)
                    dchi_per_molecule = 12 * math.pi * (D2_i / (T_ref ** 2)) * 1e-6
                    dchi_m3_per_mol = dchi_per_molecule * const.N_A
                else:
                    D2_i = D2_se_i = float("nan")
                    dchi_per_molecule = float("nan")
                    dchi_m3_per_mol = float("nan")

                self.linear_results[nuc] = {
                    "Gi": Gi,
                    "a1": a1, "sa1": sa1,
                    "b1": b1, "sb1": sb1,
                    "contact_Tref_ppm": contact_Tref,
                    "pcs_Tref_ppm": pcs_Tref,
                    "D2_i": D2_i, "D2_se_i": D2_se_i,
                    "dchi_Tref_per_molecule": dchi_per_molecule,
                    "dchi_Tref_m3_per_mol": dchi_m3_per_mol,
                }

            # 가중 평균 D2 (Gi 없는 경우 건너뜀)
            valid = [v for v in self.linear_results.values()
                     if np.isfinite(v["D2_i"]) and np.isfinite(v["D2_se_i"]) and v["D2_se_i"] > 0]
            if valid:
                wsum = sum(1.0 / (v["D2_se_i"] ** 2) for v in valid)
                D2_bar = sum(v["D2_i"] / (v["D2_se_i"] ** 2) for v in valid) / wsum
                D2_bar_se = math.sqrt(1.0 / wsum)
                dchi_bar_per_molecule = 12 * math.pi * (D2_bar / (T_ref ** 2)) * 1e-6
                dchi_bar_m3_per_mol = dchi_bar_per_molecule * const.N_A
                self.linear_summary = {
                    "N_used": len(valid),
                    "T_ref": T_ref,
                    "D2_weighted_mean": D2_bar,
                    "D2_weighted_se": D2_bar_se,
                    "DeltaChi_ax_Tref_per_molecule_weighted": dchi_bar_per_molecule,
                    "DeltaChi_ax_Tref_m3_per_mol_weighted": dchi_bar_m3_per_mol,
                }
                self.append_log(f"[Linear] Weighted D2 = {D2_bar:.6g} ± {D2_bar_se:.2g}")
                self.append_log(f"[Linear] Weighted Δχ_ax(T_ref) per mol = {dchi_bar_m3_per_mol:.6g}")
            else:
                self.linear_summary = None
                self.append_log("[Linear] No valid entries for weighted D2 (check Gi or uncertainties).")

            # GUI 표 + 로그
            self.tbl_linear.setModel(LinearTableModel(self.linear_results))
            self.append_log(f"[Linear fit δ = a/T + b/T^2] (T_ref={T_ref:g} K)")
            self.append_log("  Note: Interpretation b/Gi = D2 is exact for S2≈0, D3≈0 (classical Bleaney).")

            def _fmt_pm(val, se):
                if not np.isfinite(val):
                    return "--"
                if not np.isfinite(se):
                    return f"{val:.6g}±--"
                return f"{val:.6g}±{se:.2g}"

            def _fmt_num(val, fmt="{:.6g}"):
                if not np.isfinite(val):
                    return "--"
                return fmt.format(val)

            for nuc, v in sorted(self.linear_results.items(), key=lambda kv: natural_key(kv[0])):
                self.append_log(
                    f"  {nuc}: a={_fmt_pm(v['a1'], v['sa1'])}, "
                    f"b={_fmt_pm(v['b1'], v['sb1'])}, "
                    f"fcs_at_Tref={_fmt_num(v['contact_Tref_ppm'])} ppm, "
                    f"pcs_at_Tref={_fmt_num(v['pcs_Tref_ppm'])} ppm, "
                    f"D2_i={_fmt_num(v['D2_i'])}"
                )

        # --- Linear fit Cartesian OLS (PCS vs G_i, T_ref) — Gi 있을 때만 ---
        if getattr(self, "linear_results", None) and getattr(self.dataset, "has_G", False):
            self.cartesian_ols = self.compute_cartesian_ols(Tref_ui)
        else:
            self.cartesian_ols = None
            if not getattr(self.dataset, "has_G", False):
                self.append_log("[Info] No Gi → skip Cartesian OLS (PCS vs Gi).")

        if self.cartesian_ols:
            co = self.cartesian_ols
            self.append_log("[Cartesian OLS] PCS(T_ref) vs G_i (intercept=0)")
            self.append_log(f"  (Unweighted) slope={co['slope_ppm_per_A3_unweighted']:.6g} ppm·Å^3 "
                            f"→ Δχ = {co['DeltaChi_1e32_unweighted']:.6g} ×10^-32 m^3 "
                            f"({co['DeltaChi_m3_unweighted']:.3e} m^3)")
            self.append_log(f"  (Weighted)   slope={co['slope_ppm_per_A3_weighted']:.6g} ppm·Å^3 "
                            f"→ Δχ = {co['DeltaChi_1e32_weighted']:.6g} ×10^-32 m^3 "
                            f"({co['DeltaChi_m3_weighted']:.3e} m^3)")

        # --- Gi 없으면 baseline/extended 전부 생략 (linear-only 모드) ---
        if not getattr(self.dataset, "has_G", False):
            self.append_log("[Info] Gi missing → skipping baseline (S2=0,D3=0) and extended global fits.")
            return

        # ---------- Baseline (classical: S2=0, D3=0) ----------
        self.baseline_globals = None
        self.baseline_Fi = None
        self.baseline_diag = None
        self.baseline_metrics = None

        if self.chk_run_baseline.isChecked():
            try:
                fix_base = {"S2": 0.0, "D3": 0.0}
                if "S1" in fix_ext: fix_base["S1"] = fix_ext["S1"]
                if "D2" in fix_ext: fix_base["D2"] = fix_ext["D2"]

                base_globals, base_Fi, base_diag = fit_globals(
                    self.dataset, init, fix_base, Tref_for_diag=Tref_ui,
                    Tref_for_prior=Tref_ui, lambda_pcs_prior=lambda_pcs_prior
                )
                base_metrics = overall_metrics(self.dataset, base_globals, base_Fi)

                self.baseline_globals = base_globals
                self.baseline_Fi = base_Fi
                self.baseline_diag = base_diag
                self.baseline_metrics = base_metrics

                self.append_log("[Baseline] Classical Bleaney (S2=0, D3=0)")
                self.append_log("  Globals: " + ", ".join(f"{k}={v:.6g}" for k, v in base_globals.items()))
                self.append_log(f"  Metrics: RMSE={base_metrics['RMSE']:.6g} ppm, SSE={base_metrics['SSE']:.6g}, "
                                f"N={int(base_metrics['N_points'])}")
                tau_base = compute_tau_metrics(base_globals, Tref_ui)
                self.append_log(
                    "  Tau: " + ", ".join(f"{k}={v:.5g}" for k, v in tau_base.items() if k != "T_ref_used_K")
                    + f" (T_ref={tau_base['T_ref_used_K']:.5g} K)")
            except Exception as e:
                QMessageBox.critical(self, "Baseline fit failed", f"Least-squares failed (baseline):\n{e}")
                return

        # ---------- Extended fit ----------
        try:
            globals_out, fi_map, diag = fit_globals(
                self.dataset, init, fix_ext, Tref_for_diag=Tref_ui,
                Tref_for_prior=Tref_ui, lambda_pcs_prior=lambda_pcs_prior
            )
        except Exception as e:
            QMessageBox.critical(self, "Fit failed", f"Least-squares failed (extended):\n{e}")
            return

        ext_metrics = overall_metrics(self.dataset, globals_out, fi_map)

        self.globals_out = globals_out
        self.fi_map = fi_map
        self.diag = diag

        # update tables with extended model
        tau_ext = compute_tau_metrics(globals_out, Tref_ui)
        globals_for_table = dict(globals_out)
        globals_for_table.update(tau_ext)
        self.tbl_globals.setModel(DictTableModel(globals_for_table))
        self.tbl_diag.setModel(DiagTableModel(diag, fi_map))

        self.append_log("[Extended] Fit complete.")
        self.append_log("  Globals: " + ", ".join(f"{k}={v:.6g}" for k, v in globals_out.items()))
        self.append_log(f"  Metrics: RMSE={ext_metrics['RMSE']:.6g} ppm, SSE={ext_metrics['SSE']:.6g}, "
                        f"N={int(ext_metrics['N_points'])}")
        self.append_log("  Tau: " + ", ".join(f"{k}={v:.5g}" for k, v in tau_ext.items() if k != "T_ref_used_K")
                        + f" (T_ref={tau_ext['T_ref_used_K']:.5g} K)")

        # comparison summary if baseline exists
        if self.baseline_metrics is not None:
            d_rmse = ext_metrics["RMSE"] - self.baseline_metrics["RMSE"]
            d_sse = ext_metrics["SSE"] - self.baseline_metrics["SSE"]
            better = "Extended better" if d_rmse < 0 else "Baseline better"
            self.append_log(f"[Compare] ΔRMSE={d_rmse:.6g} ppm, ΔSSE={d_sse:.6g}  → {better}")

    def on_save_json(self):
        if not (self.dataset and self.globals_out and self.fi_map and self.diag):
            QMessageBox.warning(self, "Nothing to save", "Please run a fit first.")
            return
        suggested = "results.bleaney_fit.json"
        if self.csv_path:
            suggested = str(self.csv_path.with_suffix(".bleaney_fit.json"))
        out_path, _ = QFileDialog.getSaveFileName(self, "Save results JSON", suggested, "JSON (*.json)")
        if not out_path:
            return

        tau_ext = compute_tau_metrics(self.globals_out, float(self.ed_Tref_linear.text() or 298.0))
        tau_base = None if self.baseline_globals is None else compute_tau_metrics(self.baseline_globals, float(
            self.ed_Tref_linear.text() or 298.0))
        ext_metrics = overall_metrics(self.dataset, self.globals_out, self.fi_map)
        payload = {
            "baseline": None if self.baseline_globals is None else {
                "globals": self.baseline_globals,
                "tau": tau_base,
                "Fi": self.baseline_Fi,
                "diagnostics": self.baseline_diag,
                "metrics": self.baseline_metrics,
                "notes": "Classical Bleaney baseline: S2=0, D3=0"
            },
            "extended": {
                "globals": self.globals_out,
                "tau": tau_ext,
                "Fi": self.fi_map,
                "diagnostics": self.diag,
                "metrics": ext_metrics,
                "notes": "Model: delta = Fi*(S1/T + S2/T^2) + Gi*(D2/T^2 + D3/T^3)"
            }
        }
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to write JSON:\n{e}")
            return
        self.append_log(f"Saved JSON: {out_path}")

    def on_save_plots(self):
        if not (self.dataset and self.globals_out and self.fi_map):
            QMessageBox.warning(self, "Nothing to plot", "Please run a fit first.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select output folder for plots")
        if not out_dir:
            return
        out_dir = Path(out_dir)

        try:
            # Extended plots
            ext_dir = out_dir / "extended_plots"
            make_plots(self.dataset, self.globals_out, self.fi_map, ext_dir,
                       invert_x=self.chk_invertx.isChecked())

            # Baseline plots (if available)
            if self.baseline_globals and self.baseline_Fi:
                base_dir = out_dir / "baseline_plots"
                make_plots(self.dataset, self.baseline_globals, self.baseline_Fi, base_dir,
                           invert_x=self.chk_invertx.isChecked())

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plots:\n{e}")
            return

        msg = f"Saved plots into: {out_dir}/extended_plots"
        if self.baseline_globals and self.baseline_Fi:
            msg += f" and {out_dir}/baseline_plots"
        self.append_log(msg)

        # --- Linear plots (δ vs 1/T with two-term overlay) ---
        if getattr(self, "linear_results", None):
            lin_dir = out_dir / "linear_plots"
            lin_dir.mkdir(parents=True, exist_ok=True)
            try:
                Tref = float(self.ed_Tref_linear.text())
                if Tref <= 0:
                    raise ValueError
            except Exception:
                Tref = 298.0

            for nuc in sorted(self.dataset.groups.keys(), key=natural_key):
                sub = self.dataset.groups[nuc]
                T = sub["T"].to_numpy(float)
                d = sub["delta"].to_numpy(float)
                w = sub["w"].to_numpy(float) if "w" in sub.columns else np.ones_like(T)
                x = 1.0 / T

                vals = self.linear_results.get(nuc, {})
                a1 = vals.get("a1")
                b1 = vals.get("b1")

                # 없거나 NaN이면 즉석에서 2-항 WLS: δ ≈ a1*(1/T) + b1*(1/T^2) (절편 없음)
                if (a1 is None or b1 is None or
                        not np.isfinite(a1) or not np.isfinite(b1)):
                    X = np.column_stack([x, x ** 2])  # [1/T, 1/T^2]
                    W = np.diag(w.astype(float))
                    XtWX = X.T @ W @ X
                    XtWy = X.T @ W @ d
                    try:
                        beta = np.linalg.solve(XtWX, XtWy)
                    except np.linalg.LinAlgError:
                        beta = np.linalg.pinv(XtWX) @ XtWy
                    a1, b1 = float(beta[0]), float(beta[1])

                # T_ref에서의 분해 성분은 항상 즉석 계산(키 의존 X)
                contact_at_Tref = a1 * (1.0 / Tref)
                pcs_at_Tref = b1 * (1.0 / (Tref ** 2))

                # 그림 저장
                fig = Figure()
                ax = fig.add_subplot(111)
                ax.scatter(x, d, s=20, label=f"{nuc} data")
                xs = np.linspace(x.min(), x.max(), 300)
                ys = a1 * xs + b1 * (xs ** 2)
                ax.plot(xs, ys, label=f"fit: a={a1:.3g}, b={b1:.3g}")
                ax.set_xlabel("1 / T (K⁻¹)")
                ax.set_ylabel("δ (ppm)")
                ax.legend(title=f"T_ref={Tref:.3g} K\nFCS={contact_at_Tref:.3g} ppm\nPCS={pcs_at_Tref:.3g} ppm")
                fig.tight_layout()
                FigureCanvas(fig).print_png(str(lin_dir / f"{nuc}_delta_vs_invT.png"))

    def on_save_data(self):
        """Export per-nucleus data/model/residual and summaries to CSV."""
        if not (self.dataset and self.globals_out and self.fi_map):
            QMessageBox.warning(self, "Nothing to export", "Please run a fit first.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Select output folder for CSV export")
        if not out_dir:
            return
        base = Path(out_dir)

        # subfolders
        linear_dir = base / "linear"
        extended_dir = base / "extended"
        baseline_dir = base / "baseline"

        per_ext = extended_dir / "per_nucleus"
        per_base = baseline_dir / "per_nucleus"

        linear_dir.mkdir(parents=True, exist_ok=True)
        per_ext.mkdir(parents=True, exist_ok=True)

        if self.baseline_globals and self.baseline_Fi:
            per_base.mkdir(parents=True, exist_ok=True)

        # Linear regression (two-term) export
        if hasattr(self, "linear_results") and self.linear_results:
            NA = const.N_A;
            mu0 = const.mu_0
            try:
                T_ref = float(self.ed_Tref_linear.text())
            except Exception:
                T_ref = 298.0

            rows = []
            for k, v in sorted(self.linear_results.items(), key=lambda kv: natural_key(kv[0])):
                rows.append({
                    "nucleus": k,
                    "Gi_raw (A^-3)": v["Gi"],
                    "a1": v["a1"], "sa1": v["sa1"],
                    "b1": v["b1"], "sb1": v["sb1"],
                    "fcs_at_Tref": v["contact_Tref_ppm"],
                    "pcs_at_Tref": v["pcs_Tref_ppm"],
                    "D2_i": v["D2_i"], "D2_se_i": v["D2_se_i"],
                    "DeltaChi_ax_Tref_per_molecule": v["dchi_Tref_per_molecule"],
                    "DeltaChi_ax_Tref_m3_per_mol_no_mu0": v["dchi_Tref_per_molecule"] * NA,
                    "DeltaChi_ax_Tref_m3_per_mol_with_mu0": v["dchi_Tref_per_molecule"] * NA / mu0,
                })
            pd.DataFrame(rows).to_csv(linear_dir / "linear_approx_table.csv", index=False, encoding="utf-8-sig")

            if getattr(self, "linear_summary", None):
                ls = self.linear_summary
                pd.DataFrame([{
                    "N_used": ls["N_used"],
                    "T_ref": ls["T_ref"],
                    "D2_weighted_mean": ls["D2_weighted_mean"],
                    "D2_weighted_se": ls["D2_weighted_se"],
                    "DeltaChi_ax_Tref_per_molecule_weighted": ls["DeltaChi_ax_Tref_per_molecule_weighted"],
                    "DeltaChi_ax_Tref_m3_per_mol_weighted": ls["DeltaChi_ax_Tref_m3_per_mol_weighted"],
                    "DeltaChi_ax_Tref_m3_per_mol_with_mu0_weighted":
                        ls["DeltaChi_ax_Tref_per_molecule_weighted"] * NA / mu0
                }]).to_csv(linear_dir / "linear_approx_weighted_summary.csv", index=False, encoding="utf-8-sig")

        # --- Linear fit Cartesian OLS summary export ---
        if getattr(self, "cartesian_ols", None):
            pd.DataFrame([self.cartesian_ols]).to_csv(linear_dir / "cartesian_ols_summary.csv", index=False, encoding="utf-8-sig")

        # 개별 포인트(핵별) 산포도 원자료도 저장(원하면)
        if hasattr(self, "linear_results") and self.linear_results:
            try:
                T_ref = float(self.ed_Tref_linear.text())
            except Exception:
                T_ref = 298.0
            rows_cart = []
            for nuc in sorted(self.dataset.groups.keys(), key=natural_key):
                sub = self.dataset.groups[nuc]
                Gi = float(np.mean(sub["G_raw"].to_numpy(float)))
                lr = self.linear_results.get(nuc)
                if not lr:
                    continue
                pcs = lr["pcs_Tref_ppm"]
                sigma_pcs = lr["sb1"] / (T_ref ** 2)
                rows_cart.append({
                    "nucleus": nuc,
                    "Gi_raw (A^-3)": Gi,
                    "PCS_Tref_ppm": pcs,
                    "sigma_PCS_Tref_ppm": sigma_pcs
                })
            if rows_cart:
                pd.DataFrame(rows_cart).to_csv(linear_dir / "cartesian_points_PCS_vs_Gi.csv", index=False, encoding="utf-8-sig")

        # ---------- helper to export one model ----------
        def export_model(ds, globals_out, fi_map, per_folder, tag="extended"):
            S1, S2, D2, D3 = (globals_out[k] for k in ("S1", "S2", "D2", "D3"))
            for nuc in sorted(ds.groups.keys(), key=natural_key):
                sub = ds.groups[nuc]
                T = sub["T"].to_numpy(float)
                d = sub["delta"].to_numpy(float)
                G = sub["G"].to_numpy(float)
                Fi = fi_map[nuc]

                model = Fi * (S1 / T + S2 / (T ** 2)) + G * (D2 / (T ** 2) + D3 / (T ** 3))
                contact = Fi * (S1 / T + S2 / (T ** 2))
                pc = G * (D2 / (T ** 2) + D3 / (T ** 3))
                resid = d - model

                df_out = pd.DataFrame({
                    "nucleus": nuc,
                    "T_K": T,
                    "delta_para_ppm": d,
                    "Gi_scaled": G,
                    "Fi": Fi,
                    "model_ppm": model,
                    "residual_ppm": resid,
                    "contact_ppm": contact,
                    "pseudocontact_ppm": pc,
                })
                df_out.to_csv(per_folder / f"{nuc}_{tag}.csv", index=False, encoding="utf-8-sig")

            # summary tables
            model_dir = per_folder.parent

            # 1) globals + metrics
            met = overall_metrics(ds, globals_out, fi_map)
            try:
                T_ref = float(self.ed_Tref_linear.text())
            except Exception:
                T_ref = 298.0
            row = {
                "S1": globals_out["S1"], "S2": globals_out["S2"],
                "D2": globals_out["D2"], "D3": globals_out["D3"],
                "RMSE": met["RMSE"], "SSE": met["SSE"], "N_points": int(met["N_points"]),
            }
            row.update(compute_tau_metrics(globals_out, T_ref))
            pd.DataFrame([row]).to_csv(model_dir / "globals_and_metrics.csv", index=False, encoding="utf-8-sig")

            # 2) per-nucleus summary (Fi + diagnostics merged)
            fi_rows = [{"nucleus": k, "Fi": v} for k, v in fi_map.items()]
            df_fi = pd.DataFrame(fi_rows)

            # tag에 따라 diagnostics source 선택
            diag_src = getattr(self, "diag", None) if tag == "extended" else getattr(self, "baseline_diag", None)

            if diag_src:
                diag_rows = []
                for nuc, dct in diag_src.items():
                    row = {"nucleus": nuc}
                    row.update(
                        dct)  # n_points, T_min_K, T_max_K, RMSE_ppm, fcs_at_Tref_ppm, pcs_at_Tref_ppm, ref_T_K ...
                    diag_rows.append(row)
                df_diag = pd.DataFrame(diag_rows)
                df_sum = df_fi.merge(df_diag, on="nucleus", how="outer")
            else:
                df_sum = df_fi

            # --- nucleus 자연 정렬 ---
            df_sum["_sortkey"] = df_sum["nucleus"].astype(str).map(natural_key)
            df_sum = df_sum.sort_values("_sortkey").drop(columns=["_sortkey"])

            # --- 컬럼 순서 고정  ---
            preferred_order = [
                "nucleus",
                "n_points",
                "T_min_K",
                "T_max_K",
                "ref_T_K",
                "RMSE_ppm",
                "fcs_at_Tref_ppm",
                "pcs_at_Tref_ppm",
                "Fi",
            ]

            ordered_cols = [c for c in preferred_order if c in df_sum.columns]
            remaining_cols = [c for c in df_sum.columns if c not in ordered_cols]
            df_sum = df_sum[ordered_cols + remaining_cols]

            df_sum.to_csv(model_dir / "summary_per_nucleus.csv", index=False, encoding="utf-8-sig")

        # ---------- export extended ----------
        export_model(self.dataset, self.globals_out, self.fi_map, per_ext, tag="extended")

        # ---------- export baseline (if available) ----------
        if self.baseline_globals and self.baseline_Fi:
            export_model(self.dataset, self.baseline_globals, self.baseline_Fi, per_base, tag="baseline")

        # --- Save GUI summary tables ---
        # 1) Linear approximation table exactly like the GUI table
        if hasattr(self, "linear_results") and self.linear_results:
            try:
                T_ref = float(self.ed_Tref_linear.text())
            except Exception:
                T_ref = 298.0

            lin_rows = []
            for nuc, v in sorted(self.linear_results.items(), key=lambda kv: natural_key(kv[0])):
                lin_rows.append({
                    "Nucleus": nuc,
                    "a (=Fi*S1)": v["a1"],
                    "b (=Gi*D2)": v["b1"],
                    "fcs_at_Tref (ppm)": v["contact_Tref_ppm"],
                    "pcs_at_Tref (ppm)": v["pcs_Tref_ppm"],
                    "D2_i (=b/Gi)": v["D2_i"],
                    "Δχ_ax(T_ref) per mol (m^3/mol; no μ0)": v["dchi_Tref_m3_per_mol"],
                    "T_ref_used_K": T_ref
                })
            pd.DataFrame(lin_rows).to_csv(linear_dir / "linear_approx_table_for_gui.csv", index=False, encoding="utf-8-sig")

        # # 2) Diagnostics per nucleus — Extended (what you see in the right-side table)
        # if self.diag:
        #     diag_rows_ext = []
        #     for nuc in sorted(self.diag.keys(), key=natural_key):
        #         row = {"nucleus": nuc}
        #         row.update(self.diag[
        #                        nuc])  # has: n_points, T_min_K, T_max_K, RMSE_ppm, fcs_at_Tref_ppm, pcs_at_Tref_ppm, ref_T_K
        #         diag_rows_ext.append(row)
        #     pd.DataFrame(diag_rows_ext).to_csv(extended_dir / "diagnostics_per_nucleus.csv", index=False, encoding="utf-8-sig")
        #
        # # 3) Diagnostics per nucleus — Baseline (if available)
        # if getattr(self, "baseline_diag", None):
        #     diag_rows_base = []
        #     for nuc in sorted(self.baseline_diag.keys(), key=natural_key):
        #         row = {"nucleus": nuc}
        #         row.update(self.baseline_diag[nuc])
        #         diag_rows_base.append(row)
        #     pd.DataFrame(diag_rows_base).to_csv(baseline_dir / "diagnostics_per_nucleus.csv", index=False, encoding="utf-8-sig")

        self.append_log(f"Saved CSV data into: {out_dir}")

    def on_save_dchi(self):
        """
        Export Δχ_ax(T) as CSV for a user-defined T grid using current globals.
        - Extended: uses (D2, D3)
        - Baseline (if available): uses (D2, D3=0)
        """
        if not self.globals_out:
            QMessageBox.warning(self, "No fit", "Please run a fit first.")
            return

        # 1) 온도 범위/스텝 입력: "start, stop, step" (예: 200, 400, 1)
        text, ok = QInputDialog.getText(self, "Temperature grid",
                                        "Enter T range as 'start, stop, step' (K):",
                                        text="200, 400, 1")
        if not ok or not text.strip():
            return
        try:
            parts = [p.strip() for p in text.split(",")]
            T_start, T_stop, T_step = float(parts[0]), float(parts[1]), float(parts[2])
            if T_step <= 0 or T_stop <= T_start:
                raise ValueError
        except Exception:
            QMessageBox.critical(self, "Invalid input", "Please enter e.g. 200, 400, 1")
            return

        # 2) 출력 폴더 선택
        out_dir = QFileDialog.getExistingDirectory(self, "Select output folder for Δχ_ax CSV")
        if not out_dir:
            return
        out_dir = Path(out_dir)

        # 3) T grid 생성
        n_steps = int((T_stop - T_start) / T_step) + 1
        T_grid = np.linspace(T_start, T_start + T_step * (n_steps - 1), n_steps)

        # 4) Extended Δχ_ax(T)
        D2_ext = float(self.globals_out["D2"])
        D3_ext = float(self.globals_out.get("D3", 0.0))
        dchi_ext = delta_chi_ax_series(T_grid, D2_ext, D3_ext)
        Xppm_ext = D2_ext / (T_grid ** 2) + D3_ext / (T_grid ** 3)  # model pc coefficient in ppm
        DeltaChi_1e32_from_X_ext = Xppm_ext * 1e26

        # --- convert to molar units ---
        NA = const.N_A
        mu0 = const.mu_0
        # 방법1: μ0 미고려
        dchi_mol_no_mu0 = dchi_ext * NA
        # 방법2: μ0 고려한 SI 정의
        dchi_mol_with_mu0 = dchi_ext * NA / mu0

        df_ext = pd.DataFrame({
            "T_K": T_grid,
            "X_pc_coeff_ppm": Xppm_ext,
            "DeltaChi_1e32_from_X": DeltaChi_1e32_from_X_ext,
            "DeltaChi_ax_per_molecule_dimless": dchi_ext,
            "DeltaChi_ax_m3_per_mol_no_mu0": dchi_mol_no_mu0,
            "DeltaChi_ax_m3_per_mol_with_mu0": dchi_mol_with_mu0
        })
        df_ext.to_csv(out_dir / "DeltaChi_ax_vs_T_extended.csv", index=False, encoding="utf-8-sig")

        # 5) Baseline Δχ_ax(T)
        if self.baseline_globals:
            D2_base = float(self.baseline_globals["D2"])
            dchi_base = delta_chi_ax_series(T_grid, D2_base, 0.0)
            Xppm_base = D2_base / (T_grid ** 2)
            DeltaChi_1e32_from_X_base = Xppm_base * 1e26
            dchi_mol_no_mu0_base = dchi_base * NA
            dchi_mol_with_mu0_base = dchi_base * NA / mu0

            df_base = pd.DataFrame({
                "T_K": T_grid,
                "X_pc_coeff_ppm": Xppm_base,
                "DeltaChi_1e32_from_X": DeltaChi_1e32_from_X_base,
                "DeltaChi_ax_per_molecule_dimless": dchi_base,
                "DeltaChi_ax_m3_per_mol_no_mu0": dchi_mol_no_mu0_base,
                "DeltaChi_ax_m3_per_mol_with_mu0": dchi_mol_with_mu0_base
            })
            df_base.to_csv(out_dir / "DeltaChi_ax_vs_T_baseline.csv", index=False, encoding="utf-8-sig")

        # 6) Linear approx model (δ = a/T + b/T^2) 기반 Δχ_ax(T)
        #    D2_lin = 가중평균 D2 (없으면 단순 평균) 사용, D3=0 가정
        if getattr(self, "linear_results", None):

            # --- 비가중 평균도 미리 계산 ---
            vals_all = [v["D2_i"] for v in self.linear_results.values()
                        if np.isfinite(v.get("D2_i", np.nan))]
            D2_unw = float("nan")
            D2_unw_se = float("nan")
            if len(vals_all) > 0:
                D2_unw = float(np.mean(vals_all))
                if len(vals_all) > 1:
                    sd = float(np.std(vals_all, ddof=1))
                    D2_unw_se = sd / math.sqrt(len(vals_all))

            # --- 가중 평균 우선 ---
            D2_lin = None
            D2_lin_se = float("nan")
            if getattr(self, "linear_summary", None) and "D2_weighted_mean" in self.linear_summary:
                D2_lin = float(self.linear_summary["D2_weighted_mean"])
                D2_lin_se = float(self.linear_summary.get("D2_weighted_se", float("nan")))
            else:
                # 가중요약 없으면 단순평균으로 대체 (기존 동작)
                if np.isfinite(D2_unw):
                    D2_lin = D2_unw
                    D2_lin_se = D2_unw_se

            df_lin = None  # <- 최종 테이블 핸들

            # ---- 1) 가중/대체 D2_lin으로 기본 테이블 생성 ----
            if D2_lin is not None and np.isfinite(D2_lin):
                dchi_lin = delta_chi_ax_series(T_grid, D2_lin, 0.0)  # per molecule, dimensionless
                Xppm_lin = D2_lin / (T_grid ** 2)  # ppm
                DeltaChi_1e32_from_X_lin = Xppm_lin * 1e26
                dchi_mol_no_mu0_lin = dchi_lin * NA
                dchi_mol_with_mu0_lin = dchi_lin * NA / mu0

                df_lin = pd.DataFrame({
                    "T_K": T_grid,
                    "X_pc_coeff_ppm": Xppm_lin,
                    "DeltaChi_1e32_from_X": DeltaChi_1e32_from_X_lin,
                    "DeltaChi_ax_per_molecule_dimless": dchi_lin,
                    "DeltaChi_ax_m3_per_mol_no_mu0": dchi_mol_no_mu0_lin,
                    "DeltaChi_ax_m3_per_mol_with_mu0": dchi_mol_with_mu0_lin,
                    "D2_linear_weighted_mean": D2_lin,
                    "D2_linear_weighted_se": D2_lin_se,
                })

                # 1σ 신뢰대(옵션)
                if np.isfinite(D2_lin_se) and D2_lin_se > 0:
                    D2_up = D2_lin + D2_lin_se
                    D2_lo = D2_lin - D2_lin_se
                    Xppm_up = D2_up / (T_grid ** 2)
                    Xppm_lo = D2_lo / (T_grid ** 2)
                    dchi_up = delta_chi_ax_series(T_grid, D2_up, 0.0)
                    dchi_lo = delta_chi_ax_series(T_grid, D2_lo, 0.0)

                    df_lin["X_pc_coeff_ppm_upper_1se"] = Xppm_up
                    df_lin["X_pc_coeff_ppm_lower_1se"] = Xppm_lo
                    df_lin["DeltaChi_ax_per_molecule_dimless_upper_1se"] = dchi_up
                    df_lin["DeltaChi_ax_per_molecule_dimless_lower_1se"] = dchi_lo
                    df_lin["DeltaChi_ax_m3_per_mol_no_mu0_upper_1se"] = dchi_up * NA
                    df_lin["DeltaChi_ax_m3_per_mol_no_mu0_lower_1se"] = dchi_lo * NA
                    df_lin["DeltaChi_ax_m3_per_mol_with_mu0_upper_1se"] = dchi_up * NA / mu0
                    df_lin["DeltaChi_ax_m3_per_mol_with_mu0_lower_1se"] = dchi_lo * NA / mu0

            # ---- 2) 비가중 평균 컬럼 추가 (가능하면) ----
            if np.isfinite(D2_unw):
                Xppm_unw = D2_unw / (T_grid ** 2)
                DeltaChi_1e32_from_X_unw = Xppm_unw * 1e26
                dchi_unw = delta_chi_ax_series(T_grid, D2_unw, 0.0)
                dchi_mol_no_mu0_unw = dchi_unw * NA
                dchi_mol_with_mu0_unw = dchi_unw * NA / mu0

                if df_lin is None:
                    # 가중이 전혀 불가했을 때: 비가중만으로 테이블 생성
                    df_lin = pd.DataFrame({
                        "T_K": T_grid,
                        "X_pc_coeff_ppm_unweighted": Xppm_unw,
                        "DeltaChi_1e32_from_X_unweighted": DeltaChi_1e32_from_X_unw,
                        "DeltaChi_ax_per_molecule_dimless_unweighted": dchi_unw,
                        "DeltaChi_ax_m3_per_mol_no_mu0_unweighted": dchi_mol_no_mu0_unw,
                        "DeltaChi_ax_m3_per_mol_with_mu0_unweighted": dchi_mol_with_mu0_unw,
                        "D2_linear_unweighted_mean": D2_unw,
                        "D2_linear_unweighted_se": D2_unw_se,
                    })
                else:
                    # 같은 파일에 컬럼 추가
                    df_lin["X_pc_coeff_ppm_unweighted"] = Xppm_unw
                    df_lin["DeltaChi_1e32_from_X_unweighted"] = DeltaChi_1e32_from_X_unw
                    df_lin["DeltaChi_ax_per_molecule_dimless_unweighted"] = dchi_unw
                    df_lin["DeltaChi_ax_m3_per_mol_no_mu0_unweighted"] = dchi_mol_no_mu0_unw
                    df_lin["DeltaChi_ax_m3_per_mol_with_mu0_unweighted"] = dchi_mol_with_mu0_unw
                    df_lin["D2_linear_unweighted_mean"] = D2_unw
                    df_lin["D2_linear_unweighted_se"] = D2_unw_se

            # ---- 3) 저장/로그 ----
            if df_lin is not None:
                df_lin.to_csv(out_dir / "DeltaChi_ax_vs_T_linear_approx.csv", index=False, encoding="utf-8-sig")
                self.append_log("[Linear approx] Saved Δχ_ax(T) CSV (weighted/unweighted).")
            else:
                self.append_log("[Linear approx] Cannot export: no valid D2 (neither weighted nor unweighted).")

        self.append_log(f"Saved Δχ_ax(T) CSV to: {out_dir}")

    # ------------- Helpers -------------
    def get_cmb_value(self, cmb: QComboBox, required: bool, name: str | None = None) -> str:
        val = cmb.currentText().strip()
        if required and (val == "" or val == "<none>"):
            raise ValueError(f"Column not set{f' for {name}' if name else ''}")
        return val

    def append_log(self, msg: str):
        self.txt_log.append(msg)

    def compute_cartesian_ols(self, T_ref: float):
        """
        수동 방식과 동일: x = Gi(Å^-3), y = PCS_i(T_ref) [ppm]
        - 단순 OLS(절편=0)와 가중 OLS(절편=0, σ_y=sb1/T_ref^2) 모두 계산
        - Δχ 변환: slope_blue = δ(ppm) / Gi(Å^-3)
            Δχ[10^-32 m^3] = slope_blue * (12π) / 1e4
            Δχ[m^3]        = Δχ[10^-32] * 1e-32
        반환 dict
        """
        if self.dataset is None or not getattr(self, "linear_results", None):
            return None

        xs, ys, sy = [], [], []  # Gi, PCS_ppm, sigma_PCS_ppm
        for nuc in sorted(self.dataset.groups.keys(), key=natural_key):
            sub = self.dataset.groups[nuc]
            Gi = float(np.mean(sub["G_raw"].to_numpy(float)))  # Å⁻³ 사용
            lr = self.linear_results.get(nuc)
            if not lr:
                continue
            pcs = lr["pcs_Tref_ppm"]                  # b1 / T_ref^2
            sigma_pcs = lr["sb1"] / (T_ref**2)        # 오차 전파
            if np.isfinite(Gi) and np.isfinite(pcs):
                xs.append(Gi); ys.append(pcs)
                sy.append(sigma_pcs if np.isfinite(sigma_pcs) and sigma_pcs>0 else np.nan)

        x = np.asarray(xs, float); y = np.asarray(ys, float); sy = np.asarray(sy, float)

        # (A) 단순 OLS (절편=0)
        m_u, se_u, dof_u = zero_intercept_ols(x, y)
        dchi32_u = m_u * (12*math.pi) / 1e4
        dchi32_se_u = se_u * (12*math.pi) / 1e4
        dchi_m3_u = dchi32_u * 1e-32
        dchi_m3_se_u = dchi32_se_u * 1e-32

        # (B) 가중 OLS (절편=0, w=1/σ_y^2)
        m_w, se_w, dof_w = zero_intercept_wls(x, y, sy)
        dchi32_w = m_w * (12*math.pi) / 1e4
        dchi32_se_w = se_w * (12*math.pi) / 1e4
        dchi_m3_w = dchi32_w * 1e-32
        dchi_m3_se_w = dchi32_se_w * 1e-32

        return {
            "N_points_used": int(x.size),
            "T_ref": float(T_ref),

            "slope_ppm_per_A3_unweighted": float(m_u),
            "slope_se_unweighted": float(se_u),
            "DeltaChi_1e32_unweighted": float(dchi32_u),   # (×10^-32 m^3)
            "DeltaChi_1e32_se_unweighted": float(dchi32_se_u),
            "DeltaChi_m3_unweighted": float(dchi_m3_u),
            "DeltaChi_m3_se_unweighted": float(dchi_m3_se_u),

            "slope_ppm_per_A3_weighted": float(m_w),
            "slope_se_weighted": float(se_w),
            "DeltaChi_1e32_weighted": float(dchi32_w),     # (×10^-32 m^3)
            "DeltaChi_1e32_se_weighted": float(dchi32_se_w),
            "DeltaChi_m3_weighted": float(dchi_m3_w),
            "DeltaChi_m3_se_weighted": float(dchi_m3_se_w),
        }

# -----------------------------
# main
# -----------------------------
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
