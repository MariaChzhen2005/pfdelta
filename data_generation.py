"""
data_generation.py — PFDelta Task 4.1 solvability dataset generator.

Loads feasible and near-infeasible power flow samples for case14 and case30
(all grid contingencies: n, n-1, n-2), runs Newton-Raphson to extract Jacobian
condition numbers as solvability labels, generates balanced infeasible samples
via progressive load scaling, then saves normalized PyG graphs with 60/20/20
train/val/test splits.

Output per sample (torch_geometric.data.Data):
    x              [N, 7]     Node features: pd, qd, pg, vm, gs, bs, bus_type
    edge_index     [2, 2E]    Bidirectional branch connectivity
    edge_attr      [2E, 8]    Branch: r, x, g_fr, b_fr, g_to, b_to, tap, shift
    y_solvability  scalar     log10(condition_number) or 15.0 if infeasible
    y_state        [N, 2]     Solved [va, vm] per bus (zeros if infeasible)
    feasible_mask  bool       True if solvable, False if infeasible

Usage:
    python data_generation.py                      # full dataset
    python data_generation.py --max-samples 100    # quick dev run
    python data_generation.py --cases case14       # single case
"""

import os
import json
import glob
import time
import argparse
import logging
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from typing import Tuple, Optional, List

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm


# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

CASES = ["case14", "case30"]
GRID_TYPES = ["n", "n-1", "n-2"]

# Task 4.1 combined budget (training pool + test pool per grid type per case)
TASK_41_BUDGET = {
    "feasible":        {"n": 18200, "n-1": 18200, "n-2": 18200},
    "near_infeasible": {"n": 2000,  "n-1": 2000,  "n-2": 2000},
}

NR_MAX_ITER = 50
NR_TOL = 1e-8
INFEASIBLE_LABEL = 15.0
SPLIT_RATIOS = (0.6, 0.2, 0.2)
SEED = 42

# κ-bin thresholds for difficulty stratification (log10 scale)
DIFFICULTY_BIN_NAMES = ["easy", "moderate", "hard", "near_infeasible", "infeasible"]
DIFFICULTY_BIN_EDGES = [0.0, 3.0, 6.0, 10.0, 15.0, float("inf")]
CONTINGENCY_MAP = {"n": 0, "n-1": 1, "n-2": 2, "n-3": 3}


def assign_difficulty_bin(y_solv: float) -> int:
    """Map log10(κ) to difficulty bin index (0..4)."""
    for i in range(len(DIFFICULTY_BIN_EDGES) - 1):
        if y_solv < DIFFICULTY_BIN_EDGES[i + 1]:
            return i
    return len(DIFFICULTY_BIN_EDGES) - 2


def _is_connected(n_nodes: int, edges: list) -> bool:
    """BFS connectivity check (avoids networkx dependency)."""
    if n_nodes <= 1:
        return True
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    for s, d in edges:
        adj[s].append(d)
        adj[d].append(s)
    visited = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for nb in adj[node]:
            if nb not in visited:
                stack.append(nb)
    return len(visited) == n_nodes


DATA_ROOT = "data"
OUTPUT_DIR = "data/processed/task4_solvability"

HF_BASE_URL = (
    "https://huggingface.co/datasets/pfdelta/pfdelta/resolve/main"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class BusData:
    """Per-bus aggregated quantities extracted from a PowerModels sample."""
    types: np.ndarray   # [N] int — 1=PQ, 2=PV, 3=Slack
    pd: np.ndarray      # [N] active demand (p.u.)
    qd: np.ndarray      # [N] reactive demand (p.u.)
    pg: np.ndarray      # [N] active generation (p.u.)
    qg: np.ndarray      # [N] reactive generation (p.u.)
    vm: np.ndarray      # [N] voltage magnitude from solution
    va: np.ndarray      # [N] voltage angle from solution (rad)
    gs: np.ndarray      # [N] bus shunt conductance (p.u.)
    bs: np.ndarray      # [N] bus shunt susceptance (p.u.)


# ============================================================================
# SECTION 2: Y_BUS CONSTRUCTION
# ============================================================================

def build_ybus(network_data: dict) -> np.ndarray:
    """
    Bus admittance matrix from PowerModels branch + shunt data.
    Standard pi-model with transformer tap ratio and phase shift.
    Branches with br_status == 0 (open) are skipped.
    """
    n_bus = len(network_data["bus"])
    Y = np.zeros((n_bus, n_bus), dtype=complex)

    for branch in network_data["branch"].values():
        if branch["br_status"] == 0:
            continue
        f = int(branch["f_bus"]) - 1
        t = int(branch["t_bus"]) - 1
        y_s = 1.0 / complex(branch["br_r"], branch["br_x"])
        y_fr = complex(branch["g_fr"], branch["b_fr"])
        y_to = complex(branch["g_to"], branch["b_to"])
        tap = branch["tap"] if branch["tap"] != 0 else 1.0
        tap_c = tap * np.exp(1j * branch["shift"])

        Y[f, f] += (y_s + y_fr) / (tap ** 2)
        Y[f, t] -= y_s / np.conj(tap_c)
        Y[t, f] -= y_s / tap_c
        Y[t, t] += y_s + y_to

    for shunt in network_data.get("shunt", {}).values():
        i = int(shunt["shunt_bus"]) - 1
        Y[i, i] += complex(shunt["gs"], shunt["bs"])

    return Y


# ============================================================================
# SECTION 3: NEWTON-RAPHSON SOLVER
# ============================================================================

def compute_jacobian(Y: np.ndarray, V: np.ndarray,
                     pvpq: np.ndarray, pq: np.ndarray) -> np.ndarray:
    """
    NR Jacobian via the MATPOWER dSbus_dV formulation.
    Returns the full J = [[dP/dθ, dP/d|V|], [dQ/dθ, dQ/d|V|]]
    sub-indexed to the unknown buses (pvpq for P, pq for Q).
    """
    I_bus = Y @ V
    diag_V = np.diag(V)
    diag_I = np.diag(I_bus)
    diag_Vn = np.diag(V / np.abs(V))

    dS_dVa = 1j * diag_V @ (np.conj(diag_I) - np.conj(Y) @ np.conj(diag_V))
    dS_dVm = diag_V @ np.conj(Y @ diag_Vn) + np.conj(diag_I) @ diag_Vn

    J11 = dS_dVa[np.ix_(pvpq, pvpq)].real  # dP/dθ
    J12 = dS_dVm[np.ix_(pvpq, pq)].real     # dP/d|V|
    J21 = dS_dVa[np.ix_(pq, pvpq)].imag     # dQ/dθ
    J22 = dS_dVm[np.ix_(pq, pq)].imag       # dQ/d|V|

    return np.block([[J11, J12], [J21, J22]])


def make_v0(bd: BusData) -> np.ndarray:
    """Initial complex voltage vector: flat start with known PV/slack setpoints."""
    n = len(bd.types)
    V0 = np.ones(n, dtype=complex)
    pv = bd.types == 2
    sl = bd.types == 3
    V0[pv] = bd.vm[pv]
    V0[sl] = bd.vm[sl] * np.exp(1j * bd.va[sl])
    return V0


def newton_raphson(
    Y: np.ndarray, S_spec: np.ndarray, V0: np.ndarray,
    bus_types: np.ndarray,
    max_iter: int = NR_MAX_ITER, tol: float = NR_TOL,
) -> Tuple[np.ndarray, bool, Optional[np.ndarray], int]:
    """
    AC power flow via Newton-Raphson.
    S_spec[i] = net complex injection at bus i = (Pg - Pd) + j(Qg - Qd).
    Returns (V_solved, converged, J_final, iterations).
    J_final is the Jacobian at the converged solution (None if failed).
    """
    pv = np.where(bus_types == 2)[0]
    pq = np.where(bus_types == 1)[0]
    pvpq = np.concatenate([pv, pq])

    if len(pvpq) == 0:
        return V0.copy(), True, np.array([[1.0]]), 0

    V = V0.copy()
    for it in range(1, max_iter + 1):
        S_calc = V * np.conj(Y @ V)
        dS = S_spec - S_calc
        mis = np.concatenate([dS[pvpq].real, dS[pq].imag])

        if np.max(np.abs(mis)) < tol:
            J = compute_jacobian(Y, V, pvpq, pq)
            return V, True, J, it

        J = compute_jacobian(Y, V, pvpq, pq)
        try:
            dx = np.linalg.solve(J, mis)
        except np.linalg.LinAlgError:
            return V, False, None, it

        if not np.all(np.isfinite(dx)):
            return V, False, None, it

        n_pvpq = len(pvpq)
        Va = np.angle(V)
        Vm = np.abs(V)
        Va[pvpq] += dx[:n_pvpq]
        Vm[pq] += dx[n_pvpq:]
        V = Vm * np.exp(1j * Va)

        if np.max(np.abs(V)) > 1e6 or not np.all(np.isfinite(V)):
            return V, False, None, it

    return V, False, None, max_iter


def condition_number(J: np.ndarray) -> float:
    """2-norm condition number κ = σ_max / σ_min via SVD."""
    s = np.linalg.svd(J, compute_uv=False)
    if s[-1] < 1e-15:
        return 1e15
    return float(s[0] / s[-1])


# ============================================================================
# SECTION 4: DATA LOADING & PARSING
# ============================================================================

def parse_network(pm_case: dict, is_cpf: bool) -> Tuple[dict, dict]:
    """
    Unified parser for both standard and CPF (continuation power flow) JSON.
    Standard format: {"network": {...}, "solution": {"solution": {...}}}
    CPF format:      {"solved_net": {...}, "lambda": ...}
    Returns (network_data, solution_data) with a consistent interface.
    """
    if is_cpf:
        return pm_case["solved_net"], pm_case["solved_net"]
    return pm_case["network"], pm_case["solution"]["solution"]


def extract_bus_data(network_data: dict, solution_data: dict) -> BusData:
    """
    Aggregate per-bus quantities from PowerModels dicts.
    Loads and shunts are summed per bus. Inactive generators are skipped.
    PV buses with zero active generation are reclassified as PQ.
    """
    n = len(network_data["bus"])
    types = np.zeros(n, dtype=int)
    pd, qd, pg, qg = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    vm, va = np.zeros(n), np.zeros(n)
    gs, bs = np.zeros(n), np.zeros(n)

    for bid_str, bus in network_data["bus"].items():
        i = int(bid_str) - 1
        types[i] = bus["bus_type"]
        sol = solution_data["bus"][bid_str]
        vm[i], va[i] = sol["vm"], sol["va"]

    for load in network_data["load"].values():
        i = int(load["load_bus"]) - 1
        pd[i] += load["pd"]
        qd[i] += load["qd"]

    for gid, gen in network_data["gen"].items():
        if gen["gen_status"] != 1:
            continue
        i = int(gen["gen_bus"]) - 1
        gen_sol = solution_data["gen"][gid]
        pg[i] += gen_sol["pg"]
        qg[i] += gen_sol["qg"]

    for sh in network_data.get("shunt", {}).values():
        i = int(sh["shunt_bus"]) - 1
        gs[i] += sh["gs"]
        bs[i] += sh["bs"]

    # PV bus with no active gen effectively becomes PQ
    for i in range(n):
        if types[i] == 2 and pg[i] == 0.0:
            types[i] = 1

    return BusData(types, pd, qd, pg, qg, vm, va, gs, bs)


def collect_sample_paths(
    data_root: str, cases: List[str], grid_types: List[str], max_per_cat: int,
) -> List[Tuple[str, bool, str, str]]:
    """
    Collect JSON file paths for feasible (raw/) and near-infeasible (nose/).
    Returns [(file_path, is_cpf, case_name, grid_type), ...].
    """
    paths: List[Tuple[str, bool, str, str]] = []

    for case in cases:
        croot = os.path.join(data_root, case)
        if not os.path.isdir(croot):
            log.warning("%s not found — skipping. "
                        "Run PFDeltaDataset to download.", croot)
            continue

        for gt in grid_types:
            # Feasible samples from raw/
            raw_dir = os.path.join(croot, gt, "raw")
            if os.path.isdir(raw_dir):
                fnames = sorted(glob.glob(
                    os.path.join(raw_dir, "sample_*.json")))
                limit = TASK_41_BUDGET["feasible"][gt]
                if 0 < max_per_cat < limit:
                    limit = max_per_cat
                for fp in fnames[:limit]:
                    paths.append((fp, False, case, gt))

            # Near-infeasible samples from nose/train + nose/test
            nose_fnames: List[str] = []
            for sub in ("train", "test"):
                sd = os.path.join(croot, gt, "nose", sub)
                if os.path.isdir(sd):
                    nose_fnames.extend(
                        sorted(glob.glob(os.path.join(sd, "*.json"))))
            limit = TASK_41_BUDGET["near_infeasible"][gt]
            if 0 < max_per_cat < limit:
                limit = max_per_cat
            for fp in nose_fnames[:limit]:
                paths.append((fp, True, case, gt))

    return paths


def ensure_data(data_root: str, case_name: str) -> bool:
    """Download a case archive from HuggingFace if not already present."""
    case_dir = os.path.join(data_root, case_name)
    if os.path.isdir(case_dir):
        return True
    log.info("Downloading %s from HuggingFace...", case_name)
    try:
        from torch_geometric.data import download_url, extract_tar
        os.makedirs(data_root, exist_ok=True)
        url = f"{HF_BASE_URL}/{case_name}.tar.gz"
        tar_path = download_url(url, data_root, log=True)
        extract_tar(tar_path, data_root)
        os.unlink(tar_path)
        log.info("Extracted %s to %s", case_name, data_root)
        return True
    except Exception as e:
        log.error("Failed to download %s: %s", case_name, e)
        return False


# ============================================================================
# SECTION 5: INFEASIBLE GENERATION
# ============================================================================

def perturb_to_infeasible(
    Y: np.ndarray, bd: BusData, rng: np.random.Generator,
    scale_lo: float = 1.5, scale_hi: float = 3.0, max_attempts: int = 10,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Progressively scale loads until NR diverges.
    Returns (pd_scaled, qd_scaled, scale_used) or None if all attempts solve.
    The topology (Y_bus) and generation dispatch remain fixed; only demand
    changes — physically representing system overloading.
    """
    V0 = make_v0(bd)
    scale = rng.uniform(scale_lo, scale_hi)

    for _ in range(max_attempts):
        pd_s = bd.pd * scale
        qd_s = bd.qd * scale
        S = (bd.pg - pd_s) + 1j * (bd.qg - qd_s)
        _, converged, _, _ = newton_raphson(Y, S, V0.copy(), bd.types)
        if not converged:
            return pd_s, qd_s, scale
        scale += 0.5

    return None


# ============================================================================
# SECTION 6: PYG GRAPH CONSTRUCTION
# ============================================================================

def build_edges(network_data: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bidirectional edges from branch data.
    For each active branch (f→t), produces two directed edges:
      forward  (f→t): [r, x, g_fr, b_fr, g_to, b_to, tap, shift]
      reverse  (t→f): [r, x, g_to, b_to, g_fr, b_fr, tap, -shift]
    Returns (edge_index [2, 2E], edge_attr [2E, 8]).
    """
    src, dst, attrs = [], [], []

    for _, br in sorted(network_data["branch"].items(),
                        key=lambda kv: int(kv[0])):
        if br["br_status"] == 0:
            continue
        f = int(br["f_bus"]) - 1
        t = int(br["t_bus"]) - 1
        tap = br["tap"] if br["tap"] != 0 else 1.0

        fwd = [br["br_r"], br["br_x"],
               br["g_fr"], br["b_fr"], br["g_to"], br["b_to"],
               tap, br["shift"]]
        rev = [br["br_r"], br["br_x"],
               br["g_to"], br["b_to"], br["g_fr"], br["b_fr"],
               tap, -br["shift"]]

        src += [f, t]
        dst += [t, f]
        attrs += [fwd, rev]

    if not src:
        return (torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, 8, dtype=torch.float32))

    return (torch.tensor([src, dst], dtype=torch.long),
            torch.tensor(attrs, dtype=torch.float32))


def make_node_features(bd: BusData) -> torch.Tensor:
    """
    [N, 7] input features known before solving:
      pd, qd    — aggregate demand
      pg        — scheduled active generation
      vm        — voltage setpoint (PV/Slack) or 1.0 flat-start default (PQ)
      gs, bs    — bus shunt admittance
      bus_type  — 1 PQ / 2 PV / 3 Slack
    """
    vm_in = bd.vm.copy()
    vm_in[bd.types == 1] = 1.0   # PQ buses have no voltage setpoint
    x = np.column_stack([
        bd.pd, bd.qd, bd.pg, vm_in, bd.gs, bd.bs,
        bd.types.astype(float),
    ])
    return torch.tensor(x, dtype=torch.float32)


# ============================================================================
# SECTION 7: PIPELINE — PROCESS, GENERATE, NORMALIZE, SPLIT, SAVE
# ============================================================================

def process_solvable(path: str, is_cpf: bool, grid_type: str = "n") -> Optional[Data]:
    """
    Load one solvable sample → run NR → extract Jacobian → label with
    y = log10(condition_number). Falls back to warm-start if flat-start
    NR diverges.
    """
    try:
        with open(path) as f:
            pm = json.load(f)
        net, sol = parse_network(pm, is_cpf)
        bd = extract_bus_data(net, sol)
        Y = build_ybus(net)
        S_spec = (bd.pg - bd.pd) + 1j * (bd.qg - bd.qd)

        # Flat start
        V0 = make_v0(bd)
        V, ok, J, _ = newton_raphson(Y, S_spec, V0, bd.types)

        # Warm-start fallback using the known solution voltages
        if not ok:
            V0_warm = bd.vm * np.exp(1j * bd.va)
            V, ok, J, _ = newton_raphson(Y, S_spec, V0_warm, bd.types)

        if ok and J is not None:
            kappa = condition_number(J)
            y_solv = float(np.log10(max(kappa, 1.0)))
        else:
            # Genuinely hard sample — assign near-ceiling label
            y_solv = float(INFEASIBLE_LABEL - 1.0)

        x = make_node_features(bd)
        ei, ea = build_edges(net)
        y_state = torch.tensor(
            np.column_stack([bd.va, bd.vm]), dtype=torch.float32)

        # Extract loading margin λ* from CPF data
        y_margin = float("nan")
        if is_cpf and "lambda" in pm:
            y_margin = float(pm["lambda"])
        contingency_order = CONTINGENCY_MAP.get(grid_type, 0)

        return Data(
            x=x, edge_index=ei, edge_attr=ea,
            y_solvability=torch.tensor(y_solv, dtype=torch.float32),
            y_state=y_state,
            feasible_mask=torch.tensor(True, dtype=torch.bool),
            y_margin=torch.tensor(y_margin, dtype=torch.float32),
            contingency_order=torch.tensor(contingency_order, dtype=torch.long),
            difficulty_bin=torch.tensor(assign_difficulty_bin(y_solv), dtype=torch.long),
        )
    except Exception as e:
        log.debug("Skipping %s: %s", path, e)
        return None


def generate_infeasible(
    path: str, is_cpf: bool, rng: np.random.Generator,
    grid_type: str = "n",
) -> Optional[Data]:
    """
    Load a solvable sample, perturb loads to infeasibility, and return
    a PyG graph labelled with y=15, y_state=zeros, feasible_mask=False.
    """
    try:
        with open(path) as f:
            pm = json.load(f)
        net, sol = parse_network(pm, is_cpf)
        bd = extract_bus_data(net, sol)
        Y = build_ybus(net)

        result = perturb_to_infeasible(Y, bd, rng)
        if result is None:
            return None
        pd_s, qd_s, _ = result

        n = len(bd.types)
        vm_in = bd.vm.copy()
        vm_in[bd.types == 1] = 1.0
        x = torch.tensor(
            np.column_stack([
                pd_s, qd_s, bd.pg, vm_in, bd.gs, bd.bs,
                bd.types.astype(float),
            ]),
            dtype=torch.float32,
        )
        ei, ea = build_edges(net)

        contingency_order = CONTINGENCY_MAP.get(grid_type, 0)
        return Data(
            x=x, edge_index=ei, edge_attr=ea,
            y_solvability=torch.tensor(
                INFEASIBLE_LABEL, dtype=torch.float32),
            y_state=torch.zeros(n, 2, dtype=torch.float32),
            feasible_mask=torch.tensor(False, dtype=torch.bool),
            y_margin=torch.tensor(0.0, dtype=torch.float32),
            contingency_order=torch.tensor(contingency_order, dtype=torch.long),
            difficulty_bin=torch.tensor(assign_difficulty_bin(INFEASIBLE_LABEL), dtype=torch.long),
        )
    except Exception as e:
        log.debug("Infeasible gen failed for %s: %s", path, e)
        return None


def normalize_and_save(all_data: List[Data], output_dir: str) -> None:
    """
    1. Shuffle all samples (fixed seed).
    2. Split into train / val / test = 60 / 20 / 20.
    3. Compute per-feature mean and std from training set only.
    4. Normalize x and edge_attr across all splits (labels untouched).
    5. Save .pt data files, normalization stats, and JSON metadata.
    """
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(all_data))
    rng.shuffle(idx)

    n = len(all_data)
    n_tr = int(n * SPLIT_RATIOS[0])
    n_va = int(n * SPLIT_RATIOS[1])

    splits = {
        "train": [all_data[i] for i in idx[:n_tr]],
        "val":   [all_data[i] for i in idx[n_tr:n_tr + n_va]],
        "test":  [all_data[i] for i in idx[n_tr + n_va:]],
    }
    log.info("Split sizes — train: %d, val: %d, test: %d",
             len(splits["train"]), len(splits["val"]), len(splits["test"]))

    # Per-feature statistics from training graphs (variable-size → concatenate)
    all_x = torch.cat([d.x for d in splits["train"]], dim=0)
    all_ea = torch.cat([d.edge_attr for d in splits["train"]], dim=0)

    x_mean, x_std = all_x.mean(0), all_x.std(0)
    ea_mean, ea_std = all_ea.mean(0), all_ea.std(0)
    x_std[x_std == 0] = 1.0
    ea_std[ea_std == 0] = 1.0

    for data_list in splits.values():
        for d in data_list:
            d.x = (d.x - x_mean) / x_std
            d.edge_attr = (d.edge_attr - ea_mean) / ea_std

    os.makedirs(output_dir, exist_ok=True)

    for name, dl in splits.items():
        p = os.path.join(output_dir, f"{name}.pt")
        torch.save(dl, p)
        log.info("Saved %-5s  (%d samples) → %s", name, len(dl), p)

    torch.save(
        {"x_mean": x_mean, "x_std": x_std,
         "edge_mean": ea_mean, "edge_std": ea_std},
        os.path.join(output_dir, "norm_stats.pt"),
    )

    n_feas = sum(1 for d in all_data if d.feasible_mask.item())
    meta = {
        "total_samples": n,
        "feasible": n_feas,
        "infeasible": n - n_feas,
        "splits": {k: len(v) for k, v in splits.items()},
        "node_features": ["pd", "qd", "pg", "vm", "gs", "bs", "bus_type"],
        "edge_features": [
            "br_r", "br_x", "g_fr", "b_fr",
            "g_to", "b_to", "tap", "shift",
        ],
        "infeasible_label": INFEASIBLE_LABEL,
        "nr_max_iter": NR_MAX_ITER,
        "nr_tol": NR_TOL,
        "split_ratios": list(SPLIT_RATIOS),
        "seed": SEED,
        "cases": CASES,
        "grid_types": GRID_TYPES,
        "difficulty_bins": DIFFICULTY_BIN_NAMES,
        "difficulty_bin_edges": DIFFICULTY_BIN_EDGES[:-1],
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved metadata → %s", meta_path)


# ============================================================================
# SECTION 8: N-3 TEST SET GENERATION
# ============================================================================

def generate_n3_test_set(
    data_root: str,
    cases: List[str],
    output_dir: str,
    norm_stats_dir: str,
    n_samples: int = 2000,
    seed: int = SEED,
) -> None:
    """
    Generate an N-3 test set by removing 3 branches from feasible n-topology
    samples.  Output is normalised with the same stats as the main dataset
    and saved as ``test.pt`` under *output_dir*.
    """
    rng = np.random.default_rng(seed)
    samples: List[Data] = []

    for case in cases:
        raw_dir = os.path.join(data_root, case, "n", "raw")
        if not os.path.isdir(raw_dir):
            log.warning("N-3: %s not found — skipping.", raw_dir)
            continue
        fnames = sorted(glob.glob(os.path.join(raw_dir, "sample_*.json")))
        rng.shuffle(fnames)

        for fp in tqdm(fnames, desc=f"N-3 {case}"):
            if len(samples) >= n_samples:
                break
            try:
                with open(fp) as f:
                    pm = json.load(f)
                net, sol = parse_network(pm, False)
                bd = extract_bus_data(net, sol)

                active_bids = [
                    bid for bid, br in net["branch"].items()
                    if br["br_status"] != 0
                ]
                if len(active_bids) < 4:
                    continue

                to_remove = list(rng.choice(active_bids, size=3, replace=False))
                net_mod = deepcopy(net)
                for bid in to_remove:
                    net_mod["branch"][bid]["br_status"] = 0

                n_bus = len(net_mod["bus"])
                edges = [
                    (int(br["f_bus"]) - 1, int(br["t_bus"]) - 1)
                    for br in net_mod["branch"].values()
                    if br["br_status"] != 0
                ]
                if not _is_connected(n_bus, edges):
                    continue

                Y = build_ybus(net_mod)
                S_spec = (bd.pg - bd.pd) + 1j * (bd.qg - bd.qd)
                V0 = make_v0(bd)
                V, ok, J, _ = newton_raphson(Y, S_spec, V0, bd.types)

                if ok and J is not None:
                    kappa = condition_number(J)
                    y_solv = float(np.log10(max(kappa, 1.0)))
                    y_state = torch.tensor(
                        np.column_stack([np.angle(V), np.abs(V)]),
                        dtype=torch.float32)
                    feasible = True
                else:
                    y_solv = INFEASIBLE_LABEL
                    y_state = torch.zeros(n_bus, 2, dtype=torch.float32)
                    feasible = False

                x = make_node_features(bd)
                ei, ea = build_edges(net_mod)

                samples.append(Data(
                    x=x, edge_index=ei, edge_attr=ea,
                    y_solvability=torch.tensor(y_solv, dtype=torch.float32),
                    y_state=y_state,
                    feasible_mask=torch.tensor(feasible, dtype=torch.bool),
                    y_margin=torch.tensor(float("nan"), dtype=torch.float32),
                    contingency_order=torch.tensor(3, dtype=torch.long),
                    difficulty_bin=torch.tensor(
                        assign_difficulty_bin(y_solv), dtype=torch.long),
                ))
            except Exception as e:
                log.debug("N-3 gen failed for %s: %s", fp, e)
                continue

    log.info("N-3 test set: %d samples (%d feasible)",
             len(samples),
             sum(1 for d in samples if d.feasible_mask.item()))

    if not samples:
        log.error("No N-3 samples generated.")
        return

    # Normalise using the training statistics from the main dataset
    ns_path = os.path.join(norm_stats_dir, "norm_stats.pt")
    if not os.path.isfile(ns_path):
        log.error("norm_stats.pt not found at %s", ns_path)
        return
    ns = torch.load(ns_path, weights_only=False)
    for d in samples:
        d.x = (d.x - ns["x_mean"]) / ns["x_std"]
        d.edge_attr = (d.edge_attr - ns["edge_mean"]) / ns["edge_std"]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "test.pt")
    torch.save(samples, out_path)
    log.info("N-3 test set saved → %s", out_path)

    # Also copy norm_stats so the model can load them
    import shutil
    shutil.copy2(ns_path, os.path.join(output_dir, "norm_stats.pt"))


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--max-samples", type=int, default=-1,
        help="Cap per (case, grid_type, feasibility) category. "
             "-1 = full Task 4.1 budget.",
    )
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--cases", nargs="+", default=None,
        help="Override case list (default: case14 case30).",
    )
    parser.add_argument(
        "--skip-infeasible", action="store_true",
        help="Skip Phase 2 (infeasible generation) for debugging.",
    )
    parser.add_argument(
        "--generate-n3", action="store_true",
        help="Generate an N-3 held-out test set and exit.",
    )
    parser.add_argument(
        "--n3-output-dir", default="data/processed/n3_test",
        help="Output directory for N-3 test set.",
    )
    parser.add_argument(
        "--n3-samples", type=int, default=2000,
        help="Number of N-3 test samples to generate.",
    )
    args = parser.parse_args()

    cases = args.cases or CASES
    t0 = time.time()

    # Download any missing case data
    for case in cases:
        ensure_data(args.data_root, case)

    # ── N-3 generation mode ───────────────────────────────────────────
    if args.generate_n3:
        log.info("=" * 60)
        log.info("N-3 TEST SET GENERATION")
        log.info("=" * 60)
        generate_n3_test_set(
            data_root=args.data_root,
            cases=cases,
            output_dir=args.n3_output_dir,
            norm_stats_dir=args.output_dir,
            n_samples=args.n3_samples,
        )
        log.info("Done in %.1fs", time.time() - t0)
        return

    # Collect file paths
    paths = collect_sample_paths(
        args.data_root, cases, GRID_TYPES, args.max_samples)
    log.info("Collected %d sample paths across %s", len(paths), cases)

    if not paths:
        log.error("No data found under %s. Exiting.", args.data_root)
        return

    # ── Phase 1: Process solvable samples ─────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 1  —  Processing %d solvable samples", len(paths))
    log.info("=" * 60)

    solvable: List[Data] = []
    skipped = 0
    for fp, cpf, case, gt in tqdm(paths, desc="Solvable"):
        d = process_solvable(fp, cpf, grid_type=gt)
        if d is not None:
            solvable.append(d)
        else:
            skipped += 1

    log.info("Solvable: %d processed, %d skipped", len(solvable), skipped)

    if not solvable:
        log.error("No solvable samples produced. Check data integrity.")
        return

    # ── Phase 2: Generate infeasible samples ──────────────────────────
    if args.skip_infeasible:
        log.info("Skipping infeasible generation (--skip-infeasible).")
        infeasible: List[Data] = []
    else:
        target = len(solvable)
        log.info("=" * 60)
        log.info("PHASE 2  —  Generating %d infeasible samples", target)
        log.info("=" * 60)

        rng = np.random.default_rng(SEED)
        infeasible = []
        idx = 0
        stale = 0
        max_stale = len(paths) * 3

        pbar = tqdm(total=target, desc="Infeasible")
        while len(infeasible) < target and stale < max_stale:
            fp, cpf, _, gt = paths[idx % len(paths)]
            d = generate_infeasible(fp, cpf, rng, grid_type=gt)
            if d is not None:
                infeasible.append(d)
                pbar.update(1)
                stale = 0
            else:
                stale += 1
            idx += 1
        pbar.close()

        log.info("Infeasible: %d generated", len(infeasible))

    # ── Phase 3: Normalize, split, save ───────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 3  —  Normalize → Split → Save")
    log.info("=" * 60)

    all_data = solvable + infeasible
    log.info("Total dataset: %d  (solvable %d + infeasible %d)",
             len(all_data), len(solvable), len(infeasible))

    normalize_and_save(all_data, args.output_dir)
    log.info("Completed in %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
