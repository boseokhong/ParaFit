# logic/orca_parse_core.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np

# ---------- Common regex ----------
RE_FLOAT = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")

@dataclass(frozen=True)
class OrcaBlock:
    """Generic extracted block from ORCA output."""
    start_line: int
    end_line: int
    header: str
    lines: list[str]


def iter_line_indices(lines: Sequence[str], pattern: re.Pattern) -> Iterator[int]:
    for i, ln in enumerate(lines):
        if pattern.search(ln):
            yield i


def find_next(lines: Sequence[str],
              start: int,
              pattern: re.Pattern,
              *,
              stop_patterns: Optional[Iterable[re.Pattern]] = None,
              max_lookahead: Optional[int] = None) -> Optional[int]:
    stop_patterns = list(stop_patterns or [])
    end = len(lines) if max_lookahead is None else min(len(lines), start + max_lookahead)
    for j in range(start, end):
        ln = lines[j]
        if any(sp.search(ln) for sp in stop_patterns):
            return None
        if pattern.search(ln):
            return j
    return None


def parse_matrix3x3_from_lines(lines: Sequence[str], start_row: int) -> np.ndarray:
    """
    Parse 3 rows with at least 3 float-like tokens each.
    Raises ValueError if parsing fails.
    """
    rows = []
    for k in range(3):
        if start_row + k >= len(lines):
            raise ValueError("Unexpected end while reading 3x3 matrix.")
        toks = lines[start_row + k].split()
        vals = []
        for t in toks:
            if RE_FLOAT.fullmatch(t):
                vals.append(float(t))
            if len(vals) == 3:
                break
        if len(vals) != 3:
            raise ValueError(f"Failed to parse 3 floats from line: {lines[start_row + k]!r}")
        rows.append(vals)
    return np.array(rows, dtype=float)


def chiT_tensor_cm3Kmol_to_chi_SI_m3mol(chiT_cm3Kmol: np.ndarray,
                                        T: float,
                                        *,
                                        use_4pi: bool = True) -> np.ndarray:
    """
    chiT given in (cm^3*K/mol). Convert to chi(T) in SI (m^3/mol).
      chi_cm3/mol = chiT / T
      chi_SI      = chi_cm3/mol * 1e-6 * (4*pi if use_4pi else 1)
    """
    T = float(T)
    chi_cm3mol = np.asarray(chiT_cm3Kmol, float) / T
    chi_SI = chi_cm3mol * 1e-6
    if use_4pi:
        chi_SI *= (4.0 * np.pi)
    return chi_SI


# ---------- Specific helper: scan temperature-tagged blocks ----------
def scan_temp_tagged_tensor_blocks(text: str,
                                   *,
                                   temp_regex: re.Pattern,
                                   header_regex: re.Pattern,
                                   header_stop_regexes: Optional[Iterable[re.Pattern]] = None,
                                   use_4pi: bool = True) -> dict[float, np.ndarray]:
    """
    Generic scanner:
      - find temperature lines with temp_regex capturing group(1)=T
      - search forward for header_regex
      - read following 3 lines as 3x3
      - convert chiT->chi(T) SI using chiT_tensor_cm3Kmol_to_chi_SI_m3mol
    Returns: dict[T(K)] -> chi_SI(3x3)
    """
    lines = text.splitlines()
    stop = list(header_stop_regexes or [temp_regex])

    out: dict[float, np.ndarray] = {}
    for i in iter_line_indices(lines, temp_regex):
        m = temp_regex.search(lines[i])
        if not m:
            continue
        T = float(m.group(1))

        j = find_next(lines, i + 1, header_regex, stop_patterns=stop)
        if j is None:
            continue

        # next 3 lines after header are tensor rows
        try:
            chiT = parse_matrix3x3_from_lines(lines, j + 1)
        except ValueError:
            continue

        chi_SI = chiT_tensor_cm3Kmol_to_chi_SI_m3mol(chiT, T, use_4pi=use_4pi)
        out[T] = chi_SI

    return out
