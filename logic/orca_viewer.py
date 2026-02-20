# logic/orca_viewer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTableView, QPushButton,
    QLabel, QFileDialog, QFormLayout, QGroupBox, QLineEdit, QCheckBox, QMessageBox
)

# reuse your helpers from orca_susc
from logic.orca_susc import (
    interpolate_tensor,
    make_traceless,
    mehring_order_from_tensor,
    delta_chi_ax_rh_from_principal,
)

@dataclass
class OrcaViewerInputs:
    # ORCA χ(T) series object
    series: Any
    df: Optional[pd.DataFrame] = None
    nuc_col: Optional[str] = None
    T_col: Optional[str] = None
    G_col: Optional[str] = None


class DataFrameTableModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.df = df

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if self.df is None else len(self.df)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if self.df is None else self.df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        v = self.df.iat[index.row(), index.column()]
        if isinstance(v, float):
            return f"{v:.6g}"
        return "" if v is None else str(v)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole or self.df is None:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self.df.columns[section])
        return str(section + 1)


class OrcaChiViewer(QWidget):
    """
    Tab1: ORCA chi(T) tensor table + principal + dchi
    Tab2: (optional) PCS_orca_ppm for each CSV row given T_col & G_col
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ORCA χ(T) viewer")
        self.resize(1100, 700)

        self._inputs: Optional[OrcaViewerInputs] = None
        self._df_tensor: Optional[pd.DataFrame] = None
        self._df_pcs: Optional[pd.DataFrame] = None

        root = QVBoxLayout(self)

        # controls
        ctl_box = QGroupBox("Options")
        form = QFormLayout(ctl_box)

        self.chk_traceless = QCheckBox("Make traceless before diagonalization")
        self.chk_traceless.setChecked(True)
        self.chk_mehring = QCheckBox("Mehring order (|zz| max)")
        self.chk_mehring.setChecked(True)

        self.ed_Tref = QLineEdit("298.0")
        form.addRow(self.chk_traceless)
        form.addRow(self.chk_mehring)
        form.addRow("T_ref (K) for quick summary", self.ed_Tref)

        btn_row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_save_tensor = QPushButton("Save Tab1 CSV…")
        self.btn_save_pcs = QPushButton("Save Tab2 CSV…")
        btn_row.addWidget(self.btn_refresh)
        btn_row.addWidget(self.btn_save_tensor)
        btn_row.addWidget(self.btn_save_pcs)

        root.addWidget(ctl_box)
        root.addLayout(btn_row)

        # tabs
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        # tab1
        self.tab1 = QWidget()
        t1lay = QVBoxLayout(self.tab1)
        self.lb_summary = QLabel("")
        self.lb_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.tbl_tensor = QTableView()
        self.tbl_tensor.setSortingEnabled(True)
        t1lay.addWidget(self.lb_summary)
        t1lay.addWidget(self.tbl_tensor, 1)
        self.tabs.addTab(self.tab1, "χ(T) tensors")

        # tab2
        self.tab2 = QWidget()
        t2lay = QVBoxLayout(self.tab2)
        self.lb_pcs = QLabel("Load CSV + set T_col/G_col to enable PCS table.")
        self.lb_pcs.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.tbl_pcs = QTableView()
        self.tbl_pcs.setSortingEnabled(True)
        t2lay.addWidget(self.lb_pcs)
        t2lay.addWidget(self.tbl_pcs, 1)
        self.tabs.addTab(self.tab2, "PCS from ORCA (row-wise)")

        # signals
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_save_tensor.clicked.connect(lambda: self._save_df(self._df_tensor, "orca_chi_table.csv"))
        self.btn_save_pcs.clicked.connect(lambda: self._save_df(self._df_pcs, "orca_pcs_table.csv"))
        self.chk_traceless.stateChanged.connect(lambda *_: self.refresh())
        self.chk_mehring.stateChanged.connect(lambda *_: self.refresh())

    def set_inputs(self, inputs: OrcaViewerInputs):
        self._inputs = inputs
        self.refresh()

    def _get_series_dict(self) -> Dict[float, np.ndarray]:
        """
        Be tolerant: series may store dict under different names.
        Preferred (your new convention): tensors_per_molecule_m3
        Fallback: tensors_SI
        """
        s = self._inputs.series
        if hasattr(s, "tensors_per_molecule_m3"):
            return getattr(s, "tensors_per_molecule_m3")
        if hasattr(s, "tensors_SI"):
            return getattr(s, "tensors_SI")
        raise AttributeError("ChiTensorSeries has no tensors_per_molecule_m3 / tensors_SI dict.")

    def refresh(self):
        if self._inputs is None or self._inputs.series is None:
            return

        try:
            Tref = float(self.ed_Tref.text())
        except Exception:
            Tref = 298.0

        traceless = self.chk_traceless.isChecked()
        mehring = self.chk_mehring.isChecked()

        # -------- Tab1: tensor table --------
        try:
            dct = self._get_series_dict()
            Ts = sorted(dct.keys())
            rows = []
            for T in Ts:
                chi = np.array(dct[float(T)], float).reshape(3, 3)
                chi_use = make_traceless(chi) if traceless else chi

                if mehring:
                    vals, _vecs = mehring_order_from_tensor(chi_use)
                    cxx, cyy, czz = map(float, vals)
                else:
                    cxx, cyy, czz = float(chi_use[0, 0]), float(chi_use[1, 1]), float(chi_use[2, 2])

                dax, drh = delta_chi_ax_rh_from_principal(cxx, cyy, czz)

                rows.append({
                    "T_K": float(T),
                    "chi_xx_m3": float(chi_use[0, 0]),
                    "chi_xy_m3": float(chi_use[0, 1]),
                    "chi_xz_m3": float(chi_use[0, 2]),
                    "chi_yx_m3": float(chi_use[1, 0]),
                    "chi_yy_m3": float(chi_use[1, 1]),
                    "chi_yz_m3": float(chi_use[1, 2]),
                    "chi_zx_m3": float(chi_use[2, 0]),
                    "chi_zy_m3": float(chi_use[2, 1]),
                    "chi_zz_m3": float(chi_use[2, 2]),
                    "chi_principal_x_m3": cxx,
                    "chi_principal_y_m3": cyy,
                    "chi_principal_z_m3": czz,
                    "DeltaChi_ax_m3": dax,
                    "DeltaChi_rh_m3": drh,
                })

            df1 = pd.DataFrame(rows)
            self._df_tensor = df1
            self.tbl_tensor.setModel(DataFrameTableModel(df1))

            # quick summary at Tref
            chiT = interpolate_tensor(self._inputs.series, Tref)
            chiT = make_traceless(chiT) if traceless else chiT
            vals, _ = mehring_order_from_tensor(chiT) if mehring else (np.array([chiT[0,0], chiT[1,1], chiT[2,2]]), None)
            cxx, cyy, czz = map(float, vals)
            dax, drh = delta_chi_ax_rh_from_principal(cxx, cyy, czz)

            self.lb_summary.setText(
                f"T grid: n={len(Ts)}  Tmin={Ts[0]:.2f} K  Tmax={Ts[-1]:.2f} K\n"
                f"At T_ref={Tref:.2f} K: Δχ_ax={dax:.6e} m^3,  Δχ_rh={drh:.6e} m^3 "
                f"(traceless={traceless}, mehring={mehring})"
            )

        except Exception as e:
            QMessageBox.critical(self, "ORCA viewer error", f"Failed to build Tab1 table:\n{e}")
            return

        # -------- Tab2: PCS table --------
        df = self._inputs.df
        T_col = self._inputs.T_col
        nuc_col = self._inputs.nuc_col
        G_col = self._inputs.G_col

        if df is None or not T_col or not G_col or T_col not in df.columns or G_col not in df.columns:
            self._df_pcs = None
            self.tbl_pcs.setModel(DataFrameTableModel(pd.DataFrame()))
            self.lb_pcs.setText("Tab2 inactive: provide df + valid T_col/G_col.")
            return

        has_nuc = bool(nuc_col) and (nuc_col in df.columns)

        try:
            # PCS(ppm) = (1e36/(12π)) * G(Å^-3) * Δχ_ax(m^3)   [axial only]
            pref = 1e36 / (12.0 * np.pi)

            out_rows = []
            Ts = pd.to_numeric(df[T_col], errors="coerce").to_numpy(float)
            Gs = pd.to_numeric(df[G_col], errors="coerce").to_numpy(float)
            Ns = df[nuc_col].astype(str).to_numpy() if has_nuc else None

            for i in range(len(df)):
                N = str(Ns[i]) if Ns is not None else ""
                T = float(Ts[i])
                G = float(Gs[i])
                if not np.isfinite(T) or not np.isfinite(G):
                    out_rows.append({"row": int(i), "nucleus": N, "T_K": T, "G_A-3": G, "DeltaChi_ax_m3": np.nan, "PCS_orca_ppm": np.nan})
                    continue

                chi = interpolate_tensor(self._inputs.series, T)
                chi = make_traceless(chi) if traceless else chi
                vals, _ = mehring_order_from_tensor(chi) if mehring else (np.array([chi[0,0], chi[1,1], chi[2,2]]), None)
                cxx, cyy, czz = map(float, vals)
                dax, _drh = delta_chi_ax_rh_from_principal(cxx, cyy, czz)

                pcs_ppm = pref * G * dax

                out_rows.append({
                    "row": int(i),
                    "nucleus": N,
                    "T_K": T,
                    "G_A-3": G,
                    "DeltaChi_ax_m3": dax,
                    "PCS_orca_ppm": pcs_ppm,
                })

            df2 = pd.DataFrame(out_rows)
            self._df_pcs = df2
            self.tbl_pcs.setModel(DataFrameTableModel(df2))
            self.lb_pcs.setText(
                f"PCS using axial-only: PCS(ppm) = (1e36/(12π))*G(Å^-3)*Δχ_ax(m^3). "
                f"Note: absolute values may be incorrect if the ORCA χ tensor frame "
                f"and the Gi frame are not aligned. "
                f"Rows={len(df2)}"
            )

        except Exception as e:
            QMessageBox.critical(self, "ORCA viewer error", f"Failed to build Tab2 table:\n{e}")

    def _save_df(self, df: Optional[pd.DataFrame], default_name: str):
        if df is None or df.empty:
            QMessageBox.information(self, "No data", "Nothing to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", default_name, "CSV (*.csv)")
        if not path:
            return
        try:
            df.to_csv(path, index=False)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save CSV:\n{e}")
            return
        QMessageBox.information(self, "Saved", f"Saved:\n{path}")
