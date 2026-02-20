# logic/orca_AILFT.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ----------------------------
# Data model
# ----------------------------
@dataclass
class LFEigenfunction:
    idx: int
    energy_ev: float
    energy_cm1: float
    coeffs: Dict[str, float] = field(default_factory=dict)  # basis_label -> coeff


@dataclass
class AILFTResult:
    # header / context
    configuration: Optional[str] = None          # "f3", "d8", ...
    shell_type: Optional[str] = None             # "f" or "d"
    n_ci_blocks: Optional[int] = None
    active_mo_range: Optional[Tuple[int, int]] = None
    metal_center_atom: Optional[int] = None

    # basis labels (order matters)
    basis_labels: List[str] = field(default_factory=list)

    # ligand field one-electron matrix (a.u.) in basis_labels order
    vlft_au: Optional[np.ndarray] = None

    # Slater-Condon & Racah
    slater_condon_cm1: Dict[str, float] = field(default_factory=dict)  # e.g. "F2ff" -> cm-1
    racah_cm1: Dict[str, float] = field(default_factory=dict)          # e.g. "B" -> cm-1

    # LF one-electron eigenfunctions
    lf_eigenfunctions: List[LFEigenfunction] = field(default_factory=list)

    # SOC
    soc_constant_cm1: Optional[float] = None
    soc_constants: Dict[str, float] = field(default_factory=dict)  # e.g. "ZETA_F" -> 758.74

    # keep raw block for debugging
    raw_block: Optional[str] = None

    # convenience
    def vlft_cm1(self) -> Optional[np.ndarray]:
        # 1 Eh = 219474.6313705 cm^-1
        if self.vlft_au is None:
            return None
        return self.vlft_au * 219474.6313705


# ----------------------------
# Regex patterns
# ----------------------------
_RE_AILFT_HEADER = re.compile(r"AB INITIO LIGAND FIELD THEORY", re.IGNORECASE)
_RE_CONFIG_LINE = re.compile(r"^\s*([fdgp]\d+)\s+configuration\s*$", re.IGNORECASE)
_RE_CI_BLOCKS = re.compile(r"^\s*(\d+)\s+CI blocks\s*$", re.IGNORECASE)
_RE_MO_RANGE = re.compile(r"^\s*MOs\s+(\d+)\s+to\s+(\d+)\s*$", re.IGNORECASE)
_RE_METAL_CENTER = re.compile(r"^\s*Metal/Atom center is atom\s+(\d+)\s*$", re.IGNORECASE)
_RE_ORB_PARTS = re.compile(r"^\s*Metal/Atom\s+([a-z])\-orbital parts of active orbitals\s*$", re.IGNORECASE)

_RE_AILFT_MATRIX_HEADER_F = re.compile(r"^\s*AILFT MATRIX ELEMENTS", re.IGNORECASE)
_RE_VLFT_HEADER_D = re.compile(r"^\s*Ligand field one-electron matrix\s+VLFT\s*\(a\.u\.\)\s*:", re.IGNORECASE)

_RE_SLATER_LINE = re.compile(
    r"^\s*(F\d+\w\w)\s*(?:\(.*?\))?\s*=\s*.*?=\s*([-+]?\d+(?:\.\d+)?)\s*cm\*\*\-1",
    re.IGNORECASE
)
_RE_RACAH_LINE = re.compile(
    r"^\s*([ABCD])(?:\s*\(.*?\))?\s*=\s*.*?=\s*([-+]?\d+(?:\.\d+)?)\s*cm\*\*\-1",
    re.IGNORECASE
)

_RE_LF_EIG_HEADER = re.compile(r"^\s*The ligand field one electron eigenfunctions", re.IGNORECASE)

_RE_SOC_ZETA_LINE = re.compile(r"^\s*SOC constant zeta\s*=\s*.*?=\s*([-+]?\d+(?:\.\d+)?)\s*cm\*\*\-1", re.IGNORECASE)
_RE_ZETA_TAG = re.compile(r"^\s*(ZETA_[A-Z]+)\s*=\s*([-+]?\d+(?:\.\d+)?)\s*$", re.IGNORECASE)

# matrix helpers
_RE_ORBITAL_HEADER = re.compile(r"^\s*Orbital\b", re.IGNORECASE)


# ----------------------------
# Utilities
# ----------------------------
def _find_ailft_block(text: str, max_lines: int = 2500, *, prefer_last: bool = True) -> str:
    """
    Extract chunk starting at AILFT header.
    ORCA 출력에 AILFT가 여러 번 나오는 경우가 있어서 prefer_last=True면 마지막 블록을 선택.
    """
    lines = text.splitlines()
    hit_idxs = [i for i, ln in enumerate(lines) if _RE_AILFT_HEADER.search(ln)]
    if not hit_idxs:
        raise ValueError("AILFT header ('AB INITIO LIGAND FIELD THEORY') not found in ORCA output.")
    i0 = hit_idxs[-1] if prefer_last else hit_idxs[0]
    return "\n".join(lines[i0:i0 + max_lines])


def _parse_square_matrix_blocked(lines: List[str], start_idx: int) -> Tuple[np.ndarray, List[str], int]:
    """
    Robust ORCA matrix parser supporting column-block printing.

    Expected pattern (repeated blocks):
      Orbital   f0    f+1   f-1 ...
      f0       ...
      f+1      ...
      ...

    Returns (full_matrix, full_labels, next_index).
    """
    i = start_idx
    # find first "Orbital" header
    while i < len(lines) and not _RE_ORBITAL_HEADER.match(lines[i].strip()):
        i += 1
    if i >= len(lines):
        raise ValueError("Matrix header line starting with 'Orbital' not found.")

    # first block header labels (subset)
    header = lines[i].split()
    if len(header) < 2:
        raise ValueError("Bad matrix header line (Orbital ...).")
    col_labels = header[1:]
    full_labels = list(col_labels)

    # We need to detect full dimension (n).
    # ORCA often prints full label list in first header, but if not, we can build it across blocks.
    # We'll parse blocks and expand full_labels until row count stabilizes.
    # Approach:
    #  - Parse first block, get row labels encountered -> this is n
    #  - Then keep reading subsequent "Orbital" blocks to fill missing columns
    i += 1

    # parse first block rows to get row labels & first column subset
    row_labels: List[str] = []
    rows_tmp: List[List[float]] = []
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            i += 1
            continue
        if _RE_ORBITAL_HEADER.match(ln):
            break
        parts = ln.split()
        # stop if line is too short or looks non-numeric in value fields
        if len(parts) < 2:
            break
        rlab = parts[0]
        vals = []
        ok = True
        for x in parts[1:1 + len(col_labels)]:
            try:
                vals.append(float(x))
            except Exception:
                ok = False
                break
        if not ok:
            break
        row_labels.append(rlab)
        rows_tmp.append(vals)
        i += 1

    n = len(row_labels)
    if n == 0:
        raise ValueError("Failed to parse any matrix rows after Orbital header.")

    # initialize full matrix with first block width, will expand columns as needed
    mat = np.full((n, len(col_labels)), np.nan, float)
    mat[:, :len(col_labels)] = np.array(rows_tmp, float)

    # now parse subsequent blocks (if any) and append columns
    while i < len(lines):
        # seek next Orbital header; if not found soon, stop
        while i < len(lines) and not _RE_ORBITAL_HEADER.match(lines[i].strip()):
            # heuristic stop: if we hit another major section header, break
            if "Slater" in lines[i] or "Racah" in lines[i] or "SOC" in lines[i] or _RE_LF_EIG_HEADER.search(lines[i]):
                return _finalize_blocked_matrix(mat, full_labels, row_labels), full_labels, i
            i += 1
        if i >= len(lines) or not _RE_ORBITAL_HEADER.match(lines[i].strip()):
            break

        header = lines[i].split()
        new_cols = header[1:]
        if not new_cols:
            break
        # append new columns
        old_w = mat.shape[1]
        mat2 = np.full((n, old_w + len(new_cols)), np.nan, float)
        mat2[:, :old_w] = mat
        mat = mat2
        full_labels.extend(new_cols)

        i += 1
        rcount = 0
        while i < len(lines) and rcount < n:
            ln = lines[i].strip()
            if not ln:
                i += 1
                continue
            if _RE_ORBITAL_HEADER.match(ln):
                break
            parts = ln.split()
            if len(parts) < 2:
                break
            # row label must match row_labels ordering; tolerate mismatch by sequential fill
            rlab = parts[0]
            try:
                r = row_labels.index(rlab)
            except ValueError:
                r = rcount
            vals = []
            ok = True
            for x in parts[1:1 + len(new_cols)]:
                try:
                    vals.append(float(x))
                except Exception:
                    ok = False
                    break
            if not ok:
                break
            mat[r, old_w:old_w + len(new_cols)] = np.array(vals, float)
            rcount += 1
            i += 1

        # if we didn't fill n rows, we still continue (some ORCA prints separators); but if nothing filled, stop
        if rcount == 0:
            break

        # If matrix is square and fully filled, we can stop early
        if mat.shape[1] >= n and np.isfinite(mat[:, :n]).all():
            mat = mat[:, :n]
            full_labels = full_labels[:n]
            return _finalize_blocked_matrix(mat, full_labels, row_labels), full_labels, i

    # finalize: if we have at least n columns, take first n
    if mat.shape[1] >= n:
        mat = mat[:, :n]
        full_labels = full_labels[:n]

    return _finalize_blocked_matrix(mat, full_labels, row_labels), full_labels, i


def _finalize_blocked_matrix(mat: np.ndarray, col_labels: List[str], row_labels: List[str]) -> np.ndarray:
    """
    Map rows to same order as columns if possible, else keep parsed order.
    """
    n = mat.shape[0]
    if n != mat.shape[1]:
        # best effort: try to square by trimming
        m = min(mat.shape[0], mat.shape[1])
        mat = mat[:m, :m]
        col_labels[:] = col_labels[:m]
        row_labels[:] = row_labels[:m]

    # If row labels match column labels as a permutation, reorder rows accordingly
    try:
        order = [row_labels.index(c) for c in col_labels]
        mat = mat[order, :]
    except Exception:
        pass

    # replace remaining nan with 0 (better than crashing later)
    if np.isnan(mat).any():
        mat = np.nan_to_num(mat, nan=0.0)
    return mat


def _try_parse_config_context(block_lines: List[str], res: AILFTResult) -> None:
    for ln in block_lines[:120]:
        m = _RE_CONFIG_LINE.match(ln)
        if m:
            res.configuration = m.group(1).lower()
        m = _RE_CI_BLOCKS.match(ln)
        if m:
            res.n_ci_blocks = int(m.group(1))
        m = _RE_MO_RANGE.match(ln)
        if m:
            res.active_mo_range = (int(m.group(1)), int(m.group(2)))
        m = _RE_METAL_CENTER.match(ln)
        if m:
            res.metal_center_atom = int(m.group(1))
        m = _RE_ORB_PARTS.match(ln)
        if m:
            res.shell_type = m.group(1).lower()


def _parse_lf_eigenfunctions(block_lines: List[str], res: AILFTResult) -> None:
    idx = None
    for i, ln in enumerate(block_lines):
        if _RE_LF_EIG_HEADER.search(ln):
            idx = i
            break
    if idx is None:
        return

    # find "Orbital ..." header for the table
    i = idx
    while i < len(block_lines) and not block_lines[i].strip().startswith("Orbital"):
        i += 1
    if i >= len(block_lines):
        return

    header_tokens = block_lines[i].split()

    # basis labels start AFTER the "Energy(cm-1)" field (or any token containing "cm-1"/"cm" + "1")
    basis_start = None
    for k, tok in enumerate(header_tokens):
        t = tok.lower()
        if "cm" in t and ("-1" in t or "1" in t):
            basis_start = k + 1
            break

    # fallback: if not found, assume last numeric column is energy and basis follows
    if basis_start is None:
        # typical: Orbital Energy (eV) Energy(cm-1) ...
        basis_start = 4 if len(header_tokens) >= 5 else len(header_tokens)

    basis_labels = header_tokens[basis_start:]
    if basis_labels and not res.basis_labels:
        res.basis_labels = basis_labels

    i += 1
    while i < len(block_lines):
        ln = block_lines[i].strip()
        if not ln:
            i += 1
            continue
        if ln.lower().startswith("ligand field orbitals were stored"):
            break

        parts = block_lines[i].split()
        if len(parts) < 3:
            i += 1
            continue

        try:
            idx_i = int(parts[0])
            e_ev = float(parts[1])
            e_cm1 = float(parts[2])
        except Exception:
            i += 1
            continue

        coeff_parts = parts[3:]
        coeffs: Dict[str, float] = {}

        if res.basis_labels and len(coeff_parts) >= len(res.basis_labels):
            for lab, val in zip(res.basis_labels, coeff_parts[:len(res.basis_labels)]):
                try:
                    coeffs[lab] = float(val)
                except Exception:
                    pass
        else:
            for k, val in enumerate(coeff_parts):
                try:
                    coeffs[f"c{k}"] = float(val)
                except Exception:
                    pass

        res.lf_eigenfunctions.append(
            LFEigenfunction(idx=idx_i, energy_ev=e_ev, energy_cm1=e_cm1, coeffs=coeffs)
        )
        i += 1


def _parse_slater_racah_soc(block_lines: List[str], res: AILFTResult) -> None:
    for ln in block_lines:
        m = _RE_SLATER_LINE.match(ln)
        if m:
            res.slater_condon_cm1[m.group(1)] = float(m.group(2))
            continue
        m = _RE_RACAH_LINE.match(ln)
        if m:
            res.racah_cm1[m.group(1)] = float(m.group(2))
            continue
        m = _RE_SOC_ZETA_LINE.match(ln)
        if m:
            res.soc_constant_cm1 = float(m.group(1))
            continue
        m = _RE_ZETA_TAG.match(ln)
        if m:
            res.soc_constants[m.group(1).upper()] = float(m.group(2))
            continue


# ----------------------------
# Public API
# ----------------------------
def parse_orca_ailft(text: str) -> AILFTResult:
    block = _find_ailft_block(text, prefer_last=True)
    lines = block.splitlines()

    res = AILFTResult(raw_block=block)
    _try_parse_config_context(lines, res)

    # 1) parse LF matrix
    start_idx = None
    for i, ln in enumerate(lines):
        if _RE_VLFT_HEADER_D.search(ln):
            start_idx = i
            break
    if start_idx is None:
        for i, ln in enumerate(lines):
            if _RE_AILFT_MATRIX_HEADER_F.search(ln):
                start_idx = i
                break

    if start_idx is not None:
        try:
            mat, labels, _next = _parse_square_matrix_blocked(lines, start_idx)
            res.vlft_au = mat
            if not res.basis_labels:
                res.basis_labels = labels
        except Exception:
            # matrix 못 읽어도 나머지는 계속 파싱
            pass

    # 2) Slater/Racah/SOC constants
    _parse_slater_racah_soc(lines, res)

    # 3) LF eigenfunctions table
    _parse_lf_eigenfunctions(lines, res)

    # sanity
    if (
        res.configuration is None
        and res.vlft_au is None
        and not res.slater_condon_cm1
        and not res.lf_eigenfunctions
        and res.soc_constant_cm1 is None
        and not res.soc_constants
    ):
        raise ValueError(
            "AILFT block was found but nothing could be parsed. "
            "Provide a representative AILFT snippet to tune parser."
        )

    return res
