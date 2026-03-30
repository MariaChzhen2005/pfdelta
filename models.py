import os
import sys
import time
import json
import logging
import argparse
import itertools
import hashlib
from copy import deepcopy
from dataclasses import dataclass, asdict, fields
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from tqdm import tqdm

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_IMPEDANCE_EPS = 1e-12


def setup_file_logging(log_dir: str, prefix: str = "run") -> str:
    """Sets up file logging. Removes existing FileHandlers to avoid duplicate logs in loops."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{prefix}.log")
    
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            root_logger.removeHandler(h)
            
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
        )
    )
    root_logger.addHandler(fh)
    log.info("Log file → %s", log_path)
    return log_path


# ============================================================================
# CONFIG & SIGNATURE
# ============================================================================
@dataclass
class Config:
    # Encoder
    hidden_dim: int = 128
    num_mp_layers: int = 4
    dropout: float = 0.1
    use_global_attention: bool = True
    num_attention_heads: int = 4

    # Regulariser head
    rank_k: int = 8

    # Solver
    T: int = 5
    epsilon: float = 1e-7
    max_iter_inference: int = 20
    use_adaptive_lm: bool = False
    mu_init: float = 1e-3
    mu_min: float = 1e-8
    mu_max: float = 1e6
    mu_decrease: float = 0.5
    mu_increase: float = 2.0

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs_stage1: int = 200
    epochs_stage2: int = 30
    lambda_1: float = 1.0
    lambda_2: float = 1.0
    lambda_3: float = 1e-3
    lambda_infeasibility: float = 1.0
    grad_clip: float = 1.0

    # Scheduler
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5

    # Early stopping
    early_stop_patience: int = 25

    # Mixed precision
    use_amp: bool = False

    # Infeasibility detection thresholds
    tau: float = 50.0
    stagnation_tol: float = 1e-6

    # Voltage projection
    vm_min: float = 0.5
    vm_max: float = 1.5

    # Data convention
    bidirectional_edges: bool = True

    # DataLoader
    num_workers: int = 0
    pin_memory: bool = True

    # Paths
    data_dir: str = "data/processed/task4_solvability"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    wandb_project: str = "pfdelta-bifurcation"
    seed: int = 42


def generate_run_signature(cfg: Config) -> str:
    """Generates a unique deterministic string based on hyperparameters to prevent overwriting."""
    d = asdict(cfg)
    # Exclude system/path keys that don't affect model logic
    ignore_keys = {"data_dir", "checkpoint_dir", "log_dir", "wandb_project", "num_workers", "pin_memory"}
    for k in ignore_keys:
        d.pop(k, None)
    
    # Hash the remaining config dictionary deterministically
    s = json.dumps(d, sort_keys=True)
    hash_str = hashlib.md5(s.encode('utf-8')).hexdigest()[:6]
    
    # Add a few readable important hyperparams to the front
    readable = f"hd{cfg.hidden_dim}_lr{cfg.lr}_T{cfg.T}"
    return f"{readable}_{hash_str}"


# ============================================================================
# DEFAULT GRID-SEARCH PARAMETER SPACE
# ============================================================================
# Drastically reduced to 32 combinations (2 x 2 x 2 x 2 x 2)
DEFAULT_SWEEP_GRID: Dict[str, List[Any]] = {
    "lr": [3e-4, 1e-3],
    "hidden_dim": [64, 128],
    "num_mp_layers": [2, 4],
    "T": [3, 5],
    "dropout": [0.0, 0.1],
}


# ============================================================================
# DATA HELPERS
# ============================================================================
def load_datasets(cfg: Config):
    log.info("Loading data from %s …", cfg.data_dir)
    train_data = torch.load(
        os.path.join(cfg.data_dir, "train.pt"), weights_only=False
    )
    val_data = torch.load(
        os.path.join(cfg.data_dir, "val.pt"), weights_only=False
    )
    test_data = torch.load(
        os.path.join(cfg.data_dir, "test.pt"), weights_only=False
    )
    norm_stats = torch.load(
        os.path.join(cfg.data_dir, "norm_stats.pt"), weights_only=False
    )

    loader_kw: Dict[str, Any] = dict(
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    train_loader = DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True, **loader_kw
    )
    val_loader = DataLoader(
        val_data, batch_size=cfg.batch_size, shuffle=False, **loader_kw
    )
    test_loader = DataLoader(
        test_data, batch_size=cfg.batch_size, shuffle=False, **loader_kw
    )
    return train_loader, val_loader, test_loader, norm_stats


def denormalize_node_features(
    x_norm: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor
) -> Dict[str, torch.Tensor]:
    x_raw = x_norm * x_std.to(x_norm.device) + x_mean.to(x_norm.device)
    return {
        "pd": x_raw[:, 0],
        "qd": x_raw[:, 1],
        "pg": x_raw[:, 2],
        "vm_setpoint": x_raw[:, 3],
        "gs": x_raw[:, 4],
        "bs": x_raw[:, 5],
        "bus_type": x_raw[:, 6].round().long(),
    }


def denormalize_edge_features(
    ea_norm: torch.Tensor, ea_mean: torch.Tensor, ea_std: torch.Tensor
) -> Dict[str, torch.Tensor]:
    ea_raw = ea_norm * ea_std.to(ea_norm.device) + ea_mean.to(ea_norm.device)
    return {
        "br_r": ea_raw[:, 0],
        "br_x": ea_raw[:, 1],
        "g_fr": ea_raw[:, 2],
        "b_fr": ea_raw[:, 3],
        "g_to": ea_raw[:, 4],
        "b_to": ea_raw[:, 5],
        "tap": ea_raw[:, 6],
        "shift": ea_raw[:, 7],
    }


# ============================================================================
# PHYSICS — SINGLE-GRAPH (kept for inference / debugging)
# ============================================================================
def _admittance(r: torch.Tensor, x: torch.Tensor):
    y = 1.0 / (torch.complex(r, x) + _IMPEDANCE_EPS)
    return y.real, y.imag


class PowerFlowPhysics:
    """Differentiable π-model power-flow equations (single graph)."""

    @staticmethod
    def compute_power_injections(
        va: torch.Tensor,
        vm: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr_raw: Dict[str, torch.Tensor],
        gs: torch.Tensor,
        bs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        n = va.shape[0]

        g_s, b_s = _admittance(edge_attr_raw["br_r"], edge_attr_raw["br_x"])
        g_fr = edge_attr_raw["g_fr"]
        b_fr = edge_attr_raw["b_fr"]
        g_to = edge_attr_raw["g_to"]
        b_to = edge_attr_raw["b_to"]
        tau = edge_attr_raw["tap"]
        shift = edge_attr_raw["shift"]

        v_i, v_j = vm[src], vm[dst]
        th_i, th_j = va[src], va[dst]

        cos_f = torch.cos(th_i - th_j - shift)
        sin_f = torch.sin(th_i - th_j - shift)
        cos_t = torch.cos(th_j - th_i + shift)
        sin_t = torch.sin(th_j - th_i + shift)

        vij_tau = v_i * v_j / tau

        P_from = (v_i / tau) ** 2 * (g_s + g_fr) + vij_tau * (
            -g_s * cos_f - b_s * sin_f
        )
        Q_from = -((v_i / tau) ** 2) * (b_s + b_fr) + vij_tau * (
            -g_s * sin_f + b_s * cos_f
        )

        P_to = v_j**2 * (g_s + g_to) + vij_tau * (-g_s * cos_t - b_s * sin_t)
        Q_to = -(v_j**2) * (b_s + b_to) + vij_tau * (
            -g_s * sin_t + b_s * cos_t
        )

        P_calc = torch.zeros(n, device=va.device, dtype=va.dtype)
        P_calc.scatter_add_(0, src, P_from)
        P_calc.scatter_add_(0, dst, P_to)
        P_calc = P_calc + vm**2 * gs

        Q_calc = torch.zeros(n, device=va.device, dtype=va.dtype)
        Q_calc.scatter_add_(0, src, Q_from)
        Q_calc.scatter_add_(0, dst, Q_to)
        Q_calc = Q_calc - vm**2 * bs

        return P_calc, Q_calc

    @staticmethod
    def compute_mismatch(
        va: torch.Tensor,
        vm: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr_raw: Dict[str, torch.Tensor],
        p_spec: torch.Tensor,
        q_spec: torch.Tensor,
        gs: torch.Tensor,
        bs: torch.Tensor,
        bus_type: torch.Tensor,
        vm_setpoint: torch.Tensor,
    ) -> torch.Tensor:
        n = va.shape[0]
        P_calc, Q_calc = PowerFlowPhysics.compute_power_injections(
            va, vm, edge_index, edge_attr_raw, gs, bs
        )
        F = torch.zeros(2 * n, device=va.device, dtype=va.dtype)

        pv_mask = bus_type == 2
        sl_mask = bus_type == 3

        F[:n] = p_spec - P_calc
        F[:n] = torch.where(sl_mask, va, F[:n])

        F[n:] = q_spec - Q_calc
        F[n:] = torch.where(pv_mask | sl_mask, vm - vm_setpoint, F[n:])
        return F

    @staticmethod
    def compute_mismatch_from_x(
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr_raw: Dict[str, torch.Tensor],
        p_spec: torch.Tensor,
        q_spec: torch.Tensor,
        gs: torch.Tensor,
        bs: torch.Tensor,
        bus_type: torch.Tensor,
        vm_setpoint: torch.Tensor,
    ) -> torch.Tensor:
        n = x.shape[0] // 2
        return PowerFlowPhysics.compute_mismatch(
            x[:n],
            x[n:],
            edge_index,
            edge_attr_raw,
            p_spec,
            q_spec,
            gs,
            bs,
            bus_type,
            vm_setpoint,
        )


# ============================================================================
# PHYSICS — BATCHED  (vectorised across a same-size group)
# ============================================================================
class BatchedPhysics:
    """Vectorised F(x) and J(x) for groups of graphs sharing the same n."""

    @staticmethod
    def prepare(
        g_infos: List[Dict[str, torch.Tensor]],
        n: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Pre-compute constant (x-independent) batched tensors."""
        Bg = len(g_infos)
        all_src: List[torch.Tensor] = []
        all_dst: List[torch.Tensor] = []
        all_be: List[torch.Tensor] = []
        edge_accum: Dict[str, List[torch.Tensor]] = {
            k: []
            for k in [
                "br_r",
                "br_x",
                "g_fr",
                "b_fr",
                "g_to",
                "b_to",
                "tap",
                "shift",
            ]
        }

        for b, gi in enumerate(g_infos):
            src, dst = gi["edge_index"]
            ne = src.shape[0]
            all_src.append(src)
            all_dst.append(dst)
            all_be.append(
                torch.full((ne,), b, device=device, dtype=torch.long)
            )
            for k in edge_accum:
                edge_accum[k].append(gi["edge_attr_raw"][k])

        prep: Dict[str, Any] = {
            "src": torch.cat(all_src) if all_src else torch.zeros(0, device=device, dtype=torch.long),
            "dst": torch.cat(all_dst) if all_dst else torch.zeros(0, device=device, dtype=torch.long),
            "be": torch.cat(all_be) if all_be else torch.zeros(0, device=device, dtype=torch.long),
        }
        for k, vs in edge_accum.items():
            prep[k] = torch.cat(vs) if vs else torch.zeros(0, device=device, dtype=dtype)

        g_s, b_s = _admittance(prep["br_r"], prep["br_x"])
        prep["g_s"] = g_s
        prep["b_s"] = b_s

        prep["p_spec"] = torch.stack([gi["p_spec"] for gi in g_infos])
        prep["q_spec"] = torch.stack([gi["q_spec"] for gi in g_infos])
        prep["node_gs"] = torch.stack([gi["gs"] for gi in g_infos])
        prep["node_bs"] = torch.stack([gi["bs"] for gi in g_infos])
        prep["bus_type"] = torch.stack([gi["bus_type"] for gi in g_infos])
        prep["vm_sp"] = torch.stack([gi["vm_setpoint"] for gi in g_infos])
        return prep

    @staticmethod
    def _cast_f32(
        prep: Dict[str, Any],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in prep.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                out[k] = v.float()
            else:
                out[k] = v
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def _edge_quantities(
        va: torch.Tensor,
        vm: torch.Tensor,
        prep: Dict[str, Any],
        n: int,
    ):
        """Shared edge-level intermediates for both mismatch & Jacobian."""
        be = prep["be"]
        src = prep["src"]
        dst = prep["dst"]

        g_s = prep["g_s"]
        b_s = prep["b_s"]
        g_fr = prep["g_fr"]
        b_fr = prep["b_fr"]
        g_to = prep["g_to"]
        b_to = prep["b_to"]
        tau = prep["tap"]
        shift = prep["shift"]

        v_i = vm[be, src]
        v_j = vm[be, dst]
        th_i = va[be, src]
        th_j = va[be, dst]

        angle_f = th_i - th_j - shift
        angle_t = th_j - th_i + shift
        cos_f = torch.cos(angle_f)
        sin_f = torch.sin(angle_f)
        cos_t = torch.cos(angle_t)
        sin_t = torch.sin(angle_t)

        vij_tau = v_i * v_j / tau
        vi_tau = v_i / tau
        vi_tau_sq = vi_tau * vi_tau

        A_f = g_s * cos_f + b_s * sin_f
        B_f = g_s * sin_f - b_s * cos_f
        A_t = g_s * cos_t + b_s * sin_t
        B_t = g_s * sin_t - b_s * cos_t

        return dict(
            be=be,
            src=src,
            dst=dst,
            v_i=v_i,
            v_j=v_j,
            vi_tau=vi_tau,
            vi_tau_sq=vi_tau_sq,
            vij_tau=vij_tau,
            g_s=g_s,
            b_s=b_s,
            g_fr=g_fr,
            b_fr=b_fr,
            g_to=g_to,
            b_to=b_to,
            tau=tau,
            cos_f=cos_f,
            sin_f=sin_f,
            cos_t=cos_t,
            sin_t=sin_t,
            A_f=A_f,
            B_f=B_f,
            A_t=A_t,
            B_t=B_t,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def mismatch(
        x: torch.Tensor,
        prep: Dict[str, Any],
        n: int,
    ) -> torch.Tensor:
        """F(x) for every graph. x:[Bg,2n] → [Bg,2n]."""
        Bg = x.shape[0]
        va = x[:, :n]
        vm = x[:, n:]
        eq = BatchedPhysics._edge_quantities(va, vm, prep, n)

        P_from = eq["vi_tau_sq"] * (eq["g_s"] + eq["g_fr"]) - eq["vij_tau"] * eq["A_f"]
        Q_from = -eq["vi_tau_sq"] * (eq["b_s"] + eq["b_fr"]) - eq["vij_tau"] * eq["B_f"]
        P_to = eq["v_j"] ** 2 * (eq["g_s"] + eq["g_to"]) - eq["vij_tau"] * eq["A_t"]
        Q_to = -(eq["v_j"] ** 2) * (eq["b_s"] + eq["b_to"]) - eq["vij_tau"] * eq["B_t"]

        flat_src = eq["be"] * n + eq["src"]
        flat_dst = eq["be"] * n + eq["dst"]

        P_flat = torch.zeros(Bg * n, device=x.device, dtype=x.dtype)
        P_flat = P_flat.scatter_add(0, flat_src, P_from).scatter_add(0, flat_dst, P_to)
        P_calc = P_flat.view(Bg, n) + vm**2 * prep["node_gs"]

        Q_flat = torch.zeros(Bg * n, device=x.device, dtype=x.dtype)
        Q_flat = Q_flat.scatter_add(0, flat_src, Q_from).scatter_add(0, flat_dst, Q_to)
        Q_calc = Q_flat.view(Bg, n) - vm**2 * prep["node_bs"]

        pv = prep["bus_type"] == 2
        sl = prep["bus_type"] == 3

        F_p = prep["p_spec"] - P_calc
        F_p = torch.where(sl, va, F_p)
        F_q = prep["q_spec"] - Q_calc
        F_q = torch.where(pv | sl, vm - prep["vm_sp"], F_q)
        return torch.cat([F_p, F_q], dim=1)

    # ------------------------------------------------------------------ #
    @staticmethod
    def mismatch_and_jacobian(
        x: torch.Tensor,
        prep: Dict[str, Any],
        n: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """F(x) and J(x). Returns ([Bg,2n], [Bg,2n,2n])."""
        Bg = x.shape[0]
        device = x.device
        dtype = x.dtype
        va = x[:, :n]
        vm = x[:, n:]

        eq = BatchedPhysics._edge_quantities(va, vm, prep, n)
        be = eq["be"]
        src = eq["src"]
        dst = eq["dst"]
        vij_tau = eq["vij_tau"]
        vi_tau = eq["vi_tau"]
        vi_tau_sq = eq["vi_tau_sq"]
        v_j = eq["v_j"]
        g_s = eq["g_s"]
        b_s = eq["b_s"]
        tau = eq["tau"]
        A_f, B_f, A_t, B_t = eq["A_f"], eq["B_f"], eq["A_t"], eq["B_t"]
        g_fr, b_fr, g_to, b_to = eq["g_fr"], eq["b_fr"], eq["g_to"], eq["b_to"]

        # === mismatch (reuses eq) ===
        P_from = vi_tau_sq * (g_s + g_fr) - vij_tau * A_f
        Q_from = -vi_tau_sq * (b_s + b_fr) - vij_tau * B_f
        P_to = v_j**2 * (g_s + g_to) - vij_tau * A_t
        Q_to = -(v_j**2) * (b_s + b_to) - vij_tau * B_t

        flat_src = be * n + src
        flat_dst = be * n + dst

        P_flat = torch.zeros(Bg * n, device=device, dtype=dtype)
        P_flat = P_flat.scatter_add(0, flat_src, P_from).scatter_add(
            0, flat_dst, P_to
        )
        P_calc = P_flat.view(Bg, n) + vm**2 * prep["node_gs"]

        Q_flat = torch.zeros(Bg * n, device=device, dtype=dtype)
        Q_flat = Q_flat.scatter_add(0, flat_src, Q_from).scatter_add(
            0, flat_dst, Q_to
        )
        Q_calc = Q_flat.view(Bg, n) - vm**2 * prep["node_bs"]

        pv = prep["bus_type"] == 2
        sl = prep["bus_type"] == 3

        F_p = prep["p_spec"] - P_calc
        F_p = torch.where(sl, va, F_p)
        F_q = prep["q_spec"] - Q_calc
        F_q = torch.where(pv | sl, vm - prep["vm_sp"], F_q)
        F_batch = torch.cat([F_p, F_q], dim=1)

        # === Jacobian derivatives ===
        vj_tau = v_j / tau

        dPfs_dth_s = vij_tau * B_f
        dPfs_dth_d = -vij_tau * B_f
        dPfs_dvm_s = 2.0 * vi_tau_sq / eq["v_i"].clamp(min=1e-12) * (g_s + g_fr) - vj_tau * A_f
        dPfs_dvm_d = -vi_tau * A_f

        dQfs_dth_s = -vij_tau * A_f
        dQfs_dth_d = vij_tau * A_f
        dQfs_dvm_s = -2.0 * vi_tau_sq / eq["v_i"].clamp(min=1e-12) * (b_s + b_fr) - vj_tau * B_f
        dQfs_dvm_d = -vi_tau * B_f

        dPfd_dth_d = vij_tau * B_t
        dPfd_dth_s = -vij_tau * B_t
        dPfd_dvm_d = 2.0 * v_j * (g_s + g_to) - vi_tau * A_t
        dPfd_dvm_s = -vj_tau * A_t

        dQfd_dth_d = -vij_tau * A_t
        dQfd_dth_s = vij_tau * A_t
        dQfd_dvm_d = -2.0 * v_j * (b_s + b_to) - vi_tau * B_t
        dQfd_dvm_s = -vj_tau * B_t

        nn2 = n * n
        be4 = be.repeat(4)

        def _scatter_block(rows, cols, vals):
            r = torch.cat(rows)
            c = torch.cat(cols)
            v = torch.cat(vals)
            flat = be4 * nn2 + r * n + c
            buf = torch.zeros(Bg * nn2, device=device, dtype=dtype)
            return buf.scatter_add(0, flat, v).view(Bg, n, n)

        dFP_dva = _scatter_block(
            [src, src, dst, dst],
            [src, dst, dst, src],
            [-dPfs_dth_s, -dPfs_dth_d, -dPfd_dth_d, -dPfd_dth_s],
        )
        dFP_dvm = _scatter_block(
            [src, src, dst, dst],
            [src, dst, dst, src],
            [-dPfs_dvm_s, -dPfs_dvm_d, -dPfd_dvm_d, -dPfd_dvm_s],
        ) - torch.diag_embed(2.0 * vm * prep["node_gs"])

        dFQ_dva = _scatter_block(
            [src, src, dst, dst],
            [src, dst, dst, src],
            [-dQfs_dth_s, -dQfs_dth_d, -dQfd_dth_d, -dQfd_dth_s],
        )
        dFQ_dvm = _scatter_block(
            [src, src, dst, dst],
            [src, dst, dst, src],
            [-dQfs_dvm_s, -dQfs_dvm_d, -dQfd_dvm_d, -dQfd_dvm_s],
        ) + torch.diag_embed(2.0 * vm * prep["node_bs"])

        J = torch.cat(
            [
                torch.cat([dFP_dva, dFP_dvm], dim=2),
                torch.cat([dFQ_dva, dFQ_dvm], dim=2),
            ],
            dim=1,
        )

        keep_p = (~sl).float().unsqueeze(2)
        keep_q = (~(pv | sl)).float().unsqueeze(2)
        keep = torch.cat([keep_p, keep_q], dim=1)
        J = J * keep

        ov_p = sl.float()
        ov_q = (pv | sl).float()
        J = J + torch.diag_embed(torch.cat([ov_p, ov_q], dim=1))

        return F_batch, J


# ============================================================================
# NEURAL NETWORK COMPONENTS
# ============================================================================
class GraphTransformerLayer(nn.Module):
    """Global self-attention over nodes within each graph (padded batch)."""

    def __init__(
        self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, H: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        _, counts = torch.unique_consecutive(batch, return_counts=True)
        max_n = counts.max().item()
        B = counts.shape[0]
        dev = H.device

        splits = torch.split(H, counts.tolist())
        padded = pad_sequence(splits, batch_first=True)  # [B, max_n, d]
        key_pad = (
            torch.arange(max_n, device=dev).unsqueeze(0)
            >= counts.unsqueeze(1)
        )

        attn_out, _ = self.mha(
            padded, padded, padded, key_padding_mask=key_pad
        )
        padded = self.norm1(padded + attn_out)
        padded = self.norm2(padded + self.ffn(padded))

        # vectorised unpad
        batch_idx = torch.repeat_interleave(
            torch.arange(B, device=dev), counts
        )
        offsets = torch.zeros(B, device=dev, dtype=torch.long)
        offsets[1:] = counts.cumsum(0)[:-1]
        pos_idx = torch.arange(H.shape[0], device=dev) - torch.repeat_interleave(
            offsets, counts
        )
        return padded[batch_idx, pos_idx]


class SharedEncoder(nn.Module):
    """GINEConv message-passing + optional global graph-transformer."""

    def __init__(
        self,
        node_dim: int = 7,
        edge_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_global_attention: bool = True,
        num_attention_heads: int = 4,
    ):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.drop = nn.Dropout(dropout)

        self.use_global_attention = use_global_attention
        if use_global_attention:
            self.transformer = GraphTransformerLayer(
                hidden_dim, num_attention_heads, dropout
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.node_proj(x)
        e = self.edge_proj(edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, e)
            h_new = norm(h_new)
            h_new = F_func.leaky_relu(h_new)
            h_new = self.drop(h_new)
            h = h + h_new
        if self.use_global_attention and batch is not None:
            h = self.transformer(h, batch)
        return h


class StatePredictorHead(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        H: torch.Tensor,
        bus_type: torch.Tensor,
        vm_setpoint: torch.Tensor,
    ) -> torch.Tensor:
        delta = self.mlp(H)
        n = H.shape[0]
        va_init = torch.zeros(n, device=H.device, dtype=H.dtype)
        vm_init = torch.ones(n, device=H.device, dtype=H.dtype)
        pv_or_sl = (bus_type == 2) | (bus_type == 3)
        vm_init = torch.where(pv_or_sl, vm_setpoint, vm_init)
        return torch.cat(
            [va_init + delta[:, 0], vm_init + delta[:, 1]], dim=0
        )


class RegularizerHead(nn.Module):
    def __init__(self, hidden_dim: int = 128, rank_k: int = 8):
        super().__init__()
        self.rank_k = rank_k
        self.d_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.u_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * rank_k),
        )
        self.s_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rank_k),
        )

    def forward(
        self, H: torch.Tensor, batch: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        d_raw = self.d_mlp(H)
        u_raw = self.u_mlp(H)
        h_pool = global_mean_pool(H, batch)
        s_raw = self.s_mlp(h_pool)

        d_all = F_func.softplus(d_raw)
        s_all = F_func.softplus(s_raw)

        _, counts = torch.unique_consecutive(batch, return_counts=True)
        splits_d = torch.split(d_all, counts.tolist())
        splits_u = torch.split(u_raw, counts.tolist())

        results: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for i, (d_g, u_g) in enumerate(zip(splits_d, splits_u)):
            n_g = d_g.shape[0]
            results.append(
                (
                    d_g.reshape(2 * n_g),
                    u_g.reshape(2 * n_g, self.rank_k),
                    s_all[i],
                )
            )
        return results


class InfeasibilityHead(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, H: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        w = torch.sigmoid(self.gate(H))
        h_graph = global_add_pool(H * w, batch)
        return self.classifier(h_graph).squeeze(-1)


# ============================================================================
# UNROLLED SOLVER  (uses BatchedPhysics)
# ============================================================================
def _project_voltage_batch(
    x: torch.Tensor, n: int, vm_min: float, vm_max: float
) -> torch.Tensor:
    """Clamp Vm in-batch: x [Bg, 2n]."""
    va = x[:, :n]
    vm = x[:, n:].clamp(min=vm_min, max=vm_max)
    return torch.cat([va, vm], dim=1)


class UnrolledSolver(nn.Module):
    def __init__(
        self,
        T: int = 5,
        epsilon: float = 1e-7,
        vm_min: float = 0.5,
        vm_max: float = 1.5,
    ):
        super().__init__()
        self.T = T
        self.epsilon = epsilon
        self.vm_min = vm_min
        self.vm_max = vm_max

    # --- regularised variant ------------------------------------------ #
    def forward_batch_regularised(
        self,
        x_pred_list: List[torch.Tensor],
        reg_params: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        per_graph: List[Dict[str, torch.Tensor]],
    ) -> Tuple[List[torch.Tensor], List[List[float]]]:
        B = len(x_pred_list)
        device = x_pred_list[0].device
        dtype = x_pred_list[0].dtype

        groups: Dict[int, List[int]] = {}
        for i, gi in enumerate(per_graph):
            groups.setdefault(gi["n"], []).append(i)

        x_final_out: List[Optional[torch.Tensor]] = [None] * B
        residuals_out: List[List[float]] = [[] for _ in range(B)]

        for n_nodes, indices in groups.items():
            Bg = len(indices)
            dim = 2 * n_nodes
            k = reg_params[indices[0]][1].shape[1]

            x = torch.stack([x_pred_list[i] for i in indices])
            d_b = torch.stack([reg_params[i][0] for i in indices])
            U_b = torch.stack([reg_params[i][1] for i in indices])
            s_b = torch.stack([reg_params[i][2] for i in indices])
            g_infos = [per_graph[i] for i in indices]

            D_sqrt_b = torch.diag_embed(d_b.sqrt())
            V_b = torch.diag_embed(s_b.sqrt()) @ U_b.transpose(1, 2)

            # pre-compute constant edge data (once per group)
            prep = BatchedPhysics.prepare(g_infos, n_nodes, device, dtype)
            prep_f32 = BatchedPhysics._cast_f32(prep)
            D_sqrt_f32 = D_sqrt_b.float()
            V_f32 = V_b.float()

            last_F_max: Optional[float] = None
            for _t in range(self.T):
                with torch.autocast(device_type="cuda", enabled=False), torch.autocast(device_type="cpu", enabled=False):
                    x_f32 = x.float()
                    F_batch, J_batch = BatchedPhysics.mismatch_and_jacobian(
                        x_f32, prep_f32, n_nodes
                    )
                    max_res = F_batch.detach().abs().amax(dim=1).max().item()
                    last_F_max = max_res
                    if max_res < self.epsilon:
                        break

                    A = torch.cat([J_batch, D_sqrt_f32, V_f32], dim=1)
                    b_rhs = torch.cat(
                        [
                            -F_batch,
                            torch.zeros(
                                Bg,
                                dim + k,
                                device=device,
                                dtype=torch.float32,
                            ),
                        ],
                        dim=1,
                    )
                    dx = torch.linalg.lstsq(
                        A, b_rhs.unsqueeze(-1)
                    ).solution.squeeze(-1)

                x = x + dx.to(dtype)
                x = _project_voltage_batch(x, n_nodes, self.vm_min, self.vm_max)

            for b_idx, global_idx in enumerate(indices):
                x_final_out[global_idx] = x[b_idx]
                if last_F_max is not None:
                    residuals_out[global_idx] = [
                        F_batch[b_idx].detach().abs().max().item()
                    ]

        return x_final_out, residuals_out  # type: ignore[return-value]

    # --- adaptive LM variant ----------------------------------------- #
    def forward_batch_adaptive_lm(
        self,
        x_pred_list: List[torch.Tensor],
        per_graph: List[Dict[str, torch.Tensor]],
        cfg: Config,
    ) -> Tuple[List[torch.Tensor], List[List[float]], List[float]]:
        B = len(x_pred_list)
        device = x_pred_list[0].device
        dtype = x_pred_list[0].dtype

        groups: Dict[int, List[int]] = {}
        for i, gi in enumerate(per_graph):
            groups.setdefault(gi["n"], []).append(i)

        x_final_out: List[Optional[torch.Tensor]] = [None] * B
        residuals_out: List[List[float]] = [[] for _ in range(B)]
        mu_out: List[float] = [cfg.mu_init] * B

        for n_nodes, indices in groups.items():
            Bg = len(indices)
            dim = 2 * n_nodes

            x = torch.stack([x_pred_list[i] for i in indices])
            mu = torch.full(
                (Bg,), cfg.mu_init, device=device, dtype=torch.float32
            )
            g_infos = [per_graph[i] for i in indices]

            prep = BatchedPhysics.prepare(g_infos, n_nodes, device, dtype)
            prep_f32 = BatchedPhysics._cast_f32(prep)

            eye = torch.eye(dim, device=device, dtype=torch.float32).unsqueeze(0)
            last_F_batch: Optional[torch.Tensor] = None

            for _t in range(self.T):
                with torch.autocast(device_type="cuda", enabled=False), torch.autocast(device_type="cpu", enabled=False):
                    x_f32 = x.float()
                    F_batch, J_batch = BatchedPhysics.mismatch_and_jacobian(
                        x_f32, prep_f32, n_nodes
                    )
                    last_F_batch = F_batch
                    max_res = F_batch.detach().abs().amax(dim=1).max().item()
                    if max_res < self.epsilon:
                        break

                    mu_sqrt = mu.detach().sqrt().view(Bg, 1, 1)
                    Lam_sqrt = mu_sqrt * eye.expand(Bg, -1, -1)
                    A = torch.cat([J_batch, Lam_sqrt], dim=1)
                    b_rhs = torch.cat(
                        [
                            -F_batch,
                            torch.zeros(
                                Bg, dim, device=device, dtype=torch.float32
                            ),
                        ],
                        dim=1,
                    )
                    dx = torch.linalg.lstsq(
                        A, b_rhs.unsqueeze(-1)
                    ).solution.squeeze(-1)

                x_new = x + dx.to(dtype)
                x_new = _project_voltage_batch(
                    x_new, n_nodes, self.vm_min, self.vm_max
                )

                with torch.no_grad():
                    F_new = BatchedPhysics.mismatch(
                        x_new.float(), prep_f32, n_nodes
                    )
                    new_res = F_new.abs().amax(dim=1)
                    old_res = F_batch.detach().abs().amax(dim=1)
                    improved = new_res < old_res
                    mu = torch.where(
                        improved,
                        (mu * cfg.mu_decrease).clamp(min=cfg.mu_min),
                        (mu * cfg.mu_increase).clamp(max=cfg.mu_max),
                    )

                x = torch.where(improved.unsqueeze(1), x_new, x)

            for b_idx, global_idx in enumerate(indices):
                x_final_out[global_idx] = x[b_idx]
                mu_out[global_idx] = mu[b_idx].item()
                if last_F_batch is not None:
                    residuals_out[global_idx] = [
                        last_F_batch[b_idx].detach().abs().max().item()
                    ]

        return x_final_out, residuals_out, mu_out  # type: ignore[return-value]


# ============================================================================
# MAIN MODEL
# ============================================================================
class BifurcationAwarePFSolver(nn.Module):
    def __init__(self, cfg: Config, norm_stats: Dict[str, torch.Tensor]):
        super().__init__()
        self.cfg = cfg
        self.encoder = SharedEncoder(
            node_dim=7,
            edge_dim=8,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_mp_layers,
            dropout=cfg.dropout,
            use_global_attention=cfg.use_global_attention,
            num_attention_heads=cfg.num_attention_heads,
        )
        self.state_head = StatePredictorHead(hidden_dim=cfg.hidden_dim)
        self.infeasibility_head = InfeasibilityHead(
            hidden_dim=cfg.hidden_dim, dropout=cfg.dropout
        )
        if not cfg.use_adaptive_lm:
            self.reg_head = RegularizerHead(
                hidden_dim=cfg.hidden_dim, rank_k=cfg.rank_k
            )
        else:
            self.reg_head = None

        self.solver = UnrolledSolver(
            T=cfg.T,
            epsilon=cfg.epsilon,
            vm_min=cfg.vm_min,
            vm_max=cfg.vm_max,
        )

        self.register_buffer("x_mean", norm_stats["x_mean"])
        self.register_buffer("x_std", norm_stats["x_std"])
        self.register_buffer("edge_mean", norm_stats["edge_mean"])
        self.register_buffer("edge_std", norm_stats["edge_std"])

    def _extract_per_graph(
        self, data: Batch, node_raw: Dict[str, torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
        edge_raw = denormalize_edge_features(
            data.edge_attr, self.edge_mean, self.edge_std
        )
        batch_idx = data.batch
        edge_batch = batch_idx[data.edge_index[0]]

        _, counts = torch.unique_consecutive(batch_idx, return_counts=True)
        offsets = torch.zeros(
            counts.shape[0] + 1, dtype=torch.long, device=counts.device
        )
        offsets[1:] = counts.cumsum(0)

        per_graph: List[Dict[str, torch.Tensor]] = []
        for g_idx in range(counts.shape[0]):
            lo = offsets[g_idx].item()
            hi = offsets[g_idx + 1].item()
            n_g = hi - lo

            edge_mask = edge_batch == g_idx
            ei_g = data.edge_index[:, edge_mask] - lo
            ea_g = {k: v[edge_mask] for k, v in edge_raw.items()}

            if self.cfg.bidirectional_edges:
                fwd = ei_g[0] < ei_g[1]
                ei_phys = ei_g[:, fwd]
                ea_phys = {k: v[fwd] for k, v in ea_g.items()}
            else:
                ei_phys = ei_g
                ea_phys = ea_g

            per_graph.append(
                {
                    "edge_index": ei_phys,
                    "edge_attr_raw": ea_phys,
                    "p_spec": node_raw["pg"][lo:hi] - node_raw["pd"][lo:hi],
                    "q_spec": -node_raw["qd"][lo:hi],
                    "gs": node_raw["gs"][lo:hi],
                    "bs": node_raw["bs"][lo:hi],
                    "bus_type": node_raw["bus_type"][lo:hi],
                    "vm_setpoint": node_raw["vm_setpoint"][lo:hi],
                    "n": n_g,
                }
            )
        return per_graph

    def forward(
        self, data: Batch, run_solver: bool = True
    ) -> Dict[str, object]:
        H = self.encoder(
            data.x, data.edge_index, data.edge_attr, batch=data.batch
        )
        node_raw = denormalize_node_features(data.x, self.x_mean, self.x_std)

        x_pred = self.state_head(
            H, node_raw["bus_type"], node_raw["vm_setpoint"]
        )
        infeas_logits = self.infeasibility_head(H, data.batch)

        per_graph = self._extract_per_graph(data, node_raw)
        total_nodes = data.x.shape[0]

        x_pred_list: List[torch.Tensor] = []
        offset = 0
        for gi in per_graph:
            n_g = gi["n"]
            va_g = x_pred[offset : offset + n_g]
            vm_g = x_pred[total_nodes + offset : total_nodes + offset + n_g]
            x_pred_list.append(torch.cat([va_g, vm_g]))
            offset += n_g

        reg_params: list = []
        final_mu_list: List[float] = []

        if self.cfg.use_adaptive_lm:
            if run_solver:
                x_final_list, all_residuals, final_mu_list = (
                    self.solver.forward_batch_adaptive_lm(
                        x_pred_list, per_graph, self.cfg
                    )
                )
            else:
                x_final_list = x_pred_list
                all_residuals = [[] for _ in per_graph]
        else:
            reg_params = self.reg_head(H, data.batch)
            if run_solver:
                x_final_list, all_residuals = (
                    self.solver.forward_batch_regularised(
                        x_pred_list, reg_params, per_graph
                    )
                )
            else:
                x_final_list = x_pred_list
                all_residuals = [[] for _ in per_graph]

        return {
            "x_pred": x_pred,
            "x_pred_list": x_pred_list,
            "x_final_list": x_final_list,
            "reg_params": reg_params,
            "per_graph": per_graph,
            "infeasibility_logits": infeas_logits,
            "residuals": all_residuals,
            "final_mu_list": final_mu_list,
        }


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
def loss_state(
    x_pred: torch.Tensor,
    y_state: torch.Tensor,
    feasible_mask: torch.Tensor,
    batch: torch.Tensor,
    total_nodes: int,
) -> torch.Tensor:
    va_pred = x_pred[:total_nodes]
    vm_pred = x_pred[total_nodes:]
    va_true = y_state[:, 0]
    vm_true = y_state[:, 1]

    node_feasible = feasible_mask[batch]
    n_feas = node_feasible.sum()
    if n_feas == 0:
        return x_pred.new_tensor(0.0, requires_grad=True)
    return (
        (va_pred - va_true)[node_feasible].pow(2).sum()
        + (vm_pred - vm_true)[node_feasible].pow(2).sum()
    ) / (2.0 * n_feas)


def loss_physics(
    x_final_list: List[torch.Tensor],
    per_graph: List[Dict[str, torch.Tensor]],
    feasible_mask: torch.Tensor,
) -> torch.Tensor:
    """Batched physics loss — groups feasible graphs by size."""
    device = x_final_list[0].device
    dtype = x_final_list[0].dtype
    feasible_indices = [
        i
        for i in range(len(per_graph))
        if feasible_mask[i].item()
    ]
    if not feasible_indices:
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

    groups: Dict[int, List[int]] = {}
    for i in feasible_indices:
        groups.setdefault(per_graph[i]["n"], []).append(i)

    total = torch.tensor(0.0, device=device, dtype=dtype)
    count = 0
    for n_nodes, indices in groups.items():
        x_batch = torch.stack([x_final_list[i] for i in indices])
        g_infos = [per_graph[i] for i in indices]
        prep = BatchedPhysics.prepare(g_infos, n_nodes, device, dtype)
        F_batch = BatchedPhysics.mismatch(x_batch, prep, n_nodes)
        total = total + F_batch.pow(2).sum()
        count += F_batch.numel()

    return total / max(count, 1)


def loss_regularisation(
    reg_params: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    total = torch.tensor(0.0, device=reg_params[0][0].device)
    for d_g, _u_g, s_g in reg_params:
        total = total + d_g.sum() + s_g.sum()
    return total / max(len(reg_params), 1)


def loss_infeasibility(
    logits: torch.Tensor, feasible_mask: torch.Tensor
) -> torch.Tensor:
    return F_func.binary_cross_entropy_with_logits(
        logits, feasible_mask.float()
    )


def composite_loss(
    outputs: Dict,
    data: Batch,
    cfg: Config,
    stage: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    total_nodes = data.x.shape[0]
    feasible = data.feasible_mask

    l_s = loss_state(
        outputs["x_pred"], data.y_state, feasible, data.batch, total_nodes
    )
    l_i = loss_infeasibility(outputs["infeasibility_logits"], feasible)

    metrics: Dict[str, float] = {
        "L_state": l_s.item(),
        "L_infeas": l_i.item(),
    }

    if stage == 1:
        total = cfg.lambda_1 * l_s + cfg.lambda_infeasibility * l_i
        metrics["L_total"] = total.item()
        return total, metrics

    l_p = loss_physics(
        outputs["x_final_list"], outputs["per_graph"], feasible
    )
    metrics["L_phys"] = l_p.item()

    total = (
        cfg.lambda_1 * l_s
        + cfg.lambda_2 * l_p
        + cfg.lambda_infeasibility * l_i
    )

    if not cfg.use_adaptive_lm and outputs.get("reg_params"):
        l_r = loss_regularisation(outputs["reg_params"])
        total = total + cfg.lambda_3 * l_r
        metrics["L_reg"] = l_r.item()

    metrics["L_total"] = total.item()
    return total, metrics


# ============================================================================
# TRAINER
# ============================================================================
class Trainer:
    def __init__(
        self,
        model: BifurcationAwarePFSolver,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Config,
        device: torch.device,
        silent: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.silent = silent

        # Generate unique run signature to avoid overwriting files
        self.run_sig = generate_run_signature(cfg)
        log.info("Run signature for this configuration: %s", self.run_sig)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
        )
        self.amp_enabled = cfg.use_amp and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        setup_file_logging(cfg.log_dir, prefix=f"run_{self.run_sig}")

        if _WANDB_AVAILABLE and not silent:
            wandb.init(
                project=cfg.wandb_project, config=asdict(cfg), reinit=True
            )
            wandb.watch(model, log="gradients", log_freq=100)

        self.cumulative_train_time = 0.0
        self.best_val_loss = float("inf")
        self.best_model_state: Optional[Dict] = None
        self.es_counter = 0

    def _save_checkpoint(self, epoch: int, stage: int, tag: str = ""):
        name = f"epoch_{epoch}_stage{stage}{tag}_{self.run_sig}.pt"
        path = os.path.join(self.cfg.checkpoint_dir, name)
        torch.save(
            {
                "epoch": epoch,
                "stage": stage,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "run_sig": self.run_sig,
            },
            path,
        )
        log.info("Checkpoint → %s", path)

    @torch.no_grad()
    def _validate(self, stage: int) -> Dict[str, float]:
        self.model.eval()
        running: Dict[str, float] = {}
        count = 0
        val_iter = tqdm(
            self.val_loader,
            desc="  Validating",
            leave=False,
            disable=self.silent,
            unit="batch",
        )
        for data in val_iter:
            data = data.to(self.device)
            outputs = self.model(data, run_solver=(stage == 2))
            _, metrics = composite_loss(outputs, data, self.cfg, stage)
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v
            count += 1
        return {k: v / max(count, 1) for k, v in running.items()}

    def _run_stage(self, stage: int, num_epochs: int, epoch_offset: int):
        log.info("=" * 60)
        log.info(
            "STAGE %d — %d epochs  (solver %s)",
            stage,
            num_epochs,
            "ON" if stage == 2 else "OFF",
        )
        log.info("=" * 60)

        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.es_counter = 0

        epoch_bar = tqdm(
            range(num_epochs),
            desc=f"Stage {stage}",
            unit="ep",
            disable=self.silent,
        )
        for epoch_local in epoch_bar:
            epoch_global = epoch_offset + epoch_local
            self.model.train()
            epoch_loss = 0.0
            epoch_metrics: Dict[str, float] = {}
            n_batches = 0
            t0 = time.time()

            batch_bar = tqdm(
                self.train_loader,
                desc=f"  Ep {epoch_global + 1}",
                leave=False,
                disable=self.silent,
                unit="batch",
            )
            for data in batch_bar:
                data = data.to(self.device)
                self.optimizer.zero_grad()

                with torch.autocast(device_type="cuda", enabled=False), torch.autocast(device_type="cpu", enabled=False):
                    outputs = self.model(
                        data, run_solver=(stage == 2)
                    )
                    loss, metrics = composite_loss(
                        outputs, data, self.cfg, stage
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()
                for k, v in metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
                n_batches += 1
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            t_epoch = time.time() - t0
            self.cumulative_train_time += t_epoch
            avg_loss = epoch_loss / max(n_batches, 1)
            avg_metrics = {
                k: v / max(n_batches, 1) for k, v in epoch_metrics.items()
            }

            self.scheduler.step(avg_loss)

            val_metrics = self._validate(stage)
            val_total = val_metrics.get(
                "L_total", val_metrics.get("L_state", float("inf"))
            )

            if val_total < self.best_val_loss:
                self.best_val_loss = val_total
                self.best_model_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                self.es_counter = 0
            else:
                self.es_counter += 1

            lr_now = self.optimizer.param_groups[0]["lr"]
            epoch_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                val=f"{val_total:.4f}",
                lr=f"{lr_now:.1e}",
                es=f"{self.es_counter}/{self.cfg.early_stop_patience}",
            )

            log_dict = {
                "stage": stage,
                "epoch": epoch_global,
                "train/loss": avg_loss,
                "train/time_s": t_epoch,
                "lr": lr_now,
            }
            for k, v in avg_metrics.items():
                log_dict[f"train/{k}"] = v
            for k, v in val_metrics.items():
                log_dict[f"val/{k}"] = v
            if _WANDB_AVAILABLE and wandb.run is not None:
                wandb.log(log_dict, step=epoch_global)

            summary = (
                f"loss={avg_loss:.5f}  val={val_total:.5f}  "
                f"t={t_epoch:.1f}s  lr={lr_now:.2e}  "
                f"es={self.es_counter}/{self.cfg.early_stop_patience}"
            )
            log.info(
                "Ep %d (stage %d)  %s", epoch_global + 1, stage, summary
            )

            if (epoch_global + 1) % 10 == 0:
                self._save_checkpoint(epoch_global + 1, stage)

            if self.es_counter >= self.cfg.early_stop_patience:
                log.info(
                    "Early stopping triggered at epoch %d",
                    epoch_global + 1,
                )
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            log.info(
                "Restored best model (val_loss=%.6f)", self.best_val_loss
            )

    def train(self, stages: str = "both") -> float:
        """Returns best validation loss from the last stage run."""
        t_total = time.time()
        run_s1 = stages in ("1", "both")
        run_s2 = stages in ("2", "both")

        if run_s1:
            self._run_stage(1, self.cfg.epochs_stage1, 0)
            self._save_checkpoint(
                self.cfg.epochs_stage1, 1, tag="_final"
            )

        if run_s2:
            offset = self.cfg.epochs_stage1 if run_s1 else 0
            self._run_stage(2, self.cfg.epochs_stage2, offset)

        total_time = time.time() - t_total
        log.info("Training complete in %.1fs", total_time)

        final_path = os.path.join(
            self.cfg.checkpoint_dir, f"final_model_{self.run_sig}.pt"
        )
        torch.save(self.model.state_dict(), final_path)
        log.info("Final model → %s", final_path)
        return self.best_val_loss


# ============================================================================
# INFERENCE
# ============================================================================
class InferenceEngine:
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device

    @torch.no_grad()
    def run(
        self,
        model: BifurcationAwarePFSolver,
        dataloader: DataLoader,
        silent: bool = False,
    ) -> Dict[str, float]:
        model.eval()
        orig_T = model.solver.T
        model.solver.T = self.cfg.max_iter_inference

        all_converged: List[bool] = []
        all_pred_infeasible: List[bool] = []
        all_true_feasible: List[bool] = []
        total_samples = 0
        total_time = 0.0

        loader_bar = tqdm(
            dataloader,
            desc="Inference",
            unit="batch",
            disable=silent,
        )
        for data in loader_bar:
            data = data.to(self.device)
            B_g = data.batch.unique().shape[0]
            total_samples += B_g

            t0 = time.time()
            outputs = model(data, run_solver=True)
            total_time += time.time() - t0

            infeas_logits = outputs["infeasibility_logits"]

            for i, (x_f, gi) in enumerate(
                zip(outputs["x_final_list"], outputs["per_graph"])
            ):
                F_final = PowerFlowPhysics.compute_mismatch_from_x(
                    x_f,
                    gi["edge_index"],
                    gi["edge_attr_raw"],
                    gi["p_spec"],
                    gi["q_spec"],
                    gi["gs"],
                    gi["bs"],
                    gi["bus_type"],
                    gi["vm_setpoint"],
                )
                res_inf = F_final.abs().max().item()
                converged = res_inf < self.cfg.epsilon

                residuals = outputs["residuals"][i]
                stagnated = False
                if len(residuals) >= 2:
                    stagnated = (
                        abs(residuals[-1] - residuals[-2])
                        < self.cfg.stagnation_tol
                    )

                pred_infeas_learned = infeas_logits[i].item() < 0.0

                if self.cfg.use_adaptive_lm:
                    mu_f = (
                        outputs["final_mu_list"][i]
                        if i < len(outputs.get("final_mu_list", []))
                        else self.cfg.mu_init
                    )
                    heuristic_flag = (
                        (not converged) and stagnated and (mu_f > self.cfg.tau)
                    )
                elif outputs.get("reg_params"):
                    d_g, _, s_g = outputs["reg_params"][i]
                    tr_lam = d_g.sum().item() + s_g.sum().item()
                    heuristic_flag = (
                        (not converged)
                        and stagnated
                        and (tr_lam > self.cfg.tau)
                    )
                else:
                    heuristic_flag = (not converged) and stagnated

                pred_infeasible = pred_infeas_learned or heuristic_flag

                all_converged.append(converged)
                all_pred_infeasible.append(pred_infeasible)
                all_true_feasible.append(
                    bool(data.feasible_mask[i].item())
                )

            loader_bar.set_postfix(samples=total_samples)

        conv_rate = sum(all_converged) / max(total_samples, 1)
        true_infeasible = [not f for f in all_true_feasible]
        tp = sum(
            p and t
            for p, t in zip(all_pred_infeasible, true_infeasible)
        )
        fp = sum(
            p and (not t)
            for p, t in zip(all_pred_infeasible, true_infeasible)
        )
        fn = sum(
            (not p) and t
            for p, t in zip(all_pred_infeasible, true_infeasible)
        )
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        per_sample = total_time / max(total_samples, 1)

        metrics = {
            "inference/total_time_s": total_time,
            "inference/per_sample_time_s": per_sample,
            "inference/convergence_rate": conv_rate,
            "inference/infeasibility_precision": prec,
            "inference/infeasibility_recall": rec,
            "inference/infeasibility_f1": f1,
            "inference/total_samples": total_samples,
        }
        if _WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(metrics)

        log.info(
            "Inference: %d samples  %.2fs (%.4fs/sample)  conv=%.3f  "
            "infeas P/R/F1=%.3f/%.3f/%.3f",
            total_samples,
            total_time,
            per_sample,
            conv_rate,
            prec,
            rec,
            f1,
        )

        model.solver.T = orig_T
        return metrics


# ============================================================================
# GRID-SEARCH RUNNER
# ============================================================================
def _coerce_value(field_name: str, raw: str) -> Any:
    """Cast a CLI string to the type declared in Config."""
    type_map: Dict[str, type] = {}
    for f in fields(Config):
        type_map[f.name] = f.type
    target = type_map.get(field_name, str)
    if target is bool or target == "bool":
        return raw.lower() in ("true", "1", "yes")
    if target is int or target == "int":
        return int(raw)
    if target is float or target == "float":
        return float(raw)
    return raw


def build_experiment_grid(
    sweep_spec: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    """Cartesian product of all sweep axes."""
    if not sweep_spec:
        return [{}]
    keys = sorted(sweep_spec.keys())
    values = [sweep_spec[k] for k in keys]
    experiments: List[Dict[str, Any]] = []
    for combo in itertools.product(*values):
        experiments.append(dict(zip(keys, combo)))
    return experiments


def apply_overrides(cfg: Config, overrides: Dict[str, Any]) -> Config:
    """Return a *new* Config with the given fields replaced."""
    d = asdict(cfg)
    d.update(overrides)
    return Config(**d)


class GridSearchRunner:
    """Run a sweep over hyper-parameters and collect results."""

    def __init__(
        self,
        base_cfg: Config,
        experiments: List[Dict[str, Any]],
        device: torch.device,
        results_dir: str,
        stages: str = "both",
    ):
        self.base_cfg = base_cfg
        self.experiments = experiments
        self.device = device
        self.results_dir = results_dir
        self.stages = stages
        os.makedirs(results_dir, exist_ok=True)

    def _run_one(
        self, exp_id: int, overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        cfg = apply_overrides(self.base_cfg, overrides)
        
        # Get unique signature to embed in paths
        run_sig = generate_run_signature(cfg)
        exp_folder = f"exp_{exp_id:04d}_{run_sig}"
        
        cfg.checkpoint_dir = os.path.join(self.results_dir, exp_folder, "ckpts")
        cfg.log_dir = os.path.join(self.results_dir, exp_folder, "logs")
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)

        log.info(
            "━━ Experiment %d / %d  %s",
            exp_id + 1,
            len(self.experiments),
            overrides,
        )

        seed_everything(cfg.seed)
        train_loader, val_loader, test_loader, norm_stats = load_datasets(cfg)
        model = BifurcationAwarePFSolver(cfg, norm_stats).to(self.device)

        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            cfg,
            self.device,
            silent=True,
        )
        best_val = trainer.train(stages=self.stages)

        engine = InferenceEngine(cfg, self.device)
        test_metrics = engine.run(model, test_loader, silent=True)

        result: Dict[str, Any] = {
            "exp_id": exp_id,
            "run_sig": run_sig,
            "overrides": {k: str(v) for k, v in overrides.items()},
            "best_val_loss": best_val,
        }
        result.update(test_metrics)

        result_path = os.path.join(self.results_dir, exp_folder, f"result_{run_sig}.json")
        with open(result_path, "w") as fp:
            json.dump(result, fp, indent=2)
            
        log.info(
            "  → val=%.5f  conv=%.3f  f1=%.3f",
            best_val,
            test_metrics.get("inference/convergence_rate", 0),
            test_metrics.get("inference/infeasibility_f1", 0),
        )
        return result

    def run(
        self,
        experiment_ids: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        if experiment_ids is None:
            experiment_ids = list(range(len(self.experiments)))

        all_results: List[Dict[str, Any]] = []
        exp_bar = tqdm(
            experiment_ids,
            desc="Grid Search",
            unit="exp",
        )
        for eid in exp_bar:
            overrides = self.experiments[eid]
            exp_bar.set_postfix(
                exp=eid,
                params=", ".join(f"{k}={v}" for k, v in overrides.items()),
            )
            result = self._run_one(eid, overrides)
            all_results.append(result)

        summary_path = os.path.join(self.results_dir, "summary.json")
        with open(summary_path, "w") as fp:
            json.dump(all_results, fp, indent=2)
        log.info("Grid search summary → %s", summary_path)

        if all_results:
            best = min(all_results, key=lambda r: r["best_val_loss"])
            log.info(
                "Best experiment: %d (sig: %s)  val=%.5f  %s",
                best["exp_id"],
                best.get("run_sig", ""),
                best["best_val_loss"],
                best["overrides"],
            )
        return all_results


# ============================================================================
# MAIN
# ============================================================================
def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bifurcation-Aware PF Solver",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── paths ──
    g = p.add_argument_group("Paths")
    g.add_argument("--data-dir", default="data/processed/task4_solvability")
    g.add_argument("--checkpoint-dir", default="checkpoints")
    g.add_argument("--log-dir", default="logs")
    g.add_argument("--wandb-project", default="pfdelta-bifurcation")

    # ── encoder ──
    g = p.add_argument_group("Encoder")
    g.add_argument("--hidden-dim", type=int, default=128)
    g.add_argument("--num-mp-layers", type=int, default=4)
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--no-global-attention", action="store_true")
    g.add_argument("--num-attention-heads", type=int, default=4)
    g.add_argument("--rank-k", type=int, default=8)

    # ── solver ──
    g = p.add_argument_group("Solver")
    g.add_argument("--T", type=int, default=5)
    g.add_argument("--epsilon", type=float, default=1e-7)
    g.add_argument("--max-iter-inference", type=int, default=20)
    g.add_argument("--adaptive-lm", action="store_true")
    g.add_argument("--mu-init", type=float, default=1e-3)
    g.add_argument("--mu-min", type=float, default=1e-8)
    g.add_argument("--mu-max", type=float, default=1e6)
    g.add_argument("--mu-decrease", type=float, default=0.5)
    g.add_argument("--mu-increase", type=float, default=2.0)

    # ── training ──
    g = p.add_argument_group("Training")
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight-decay", type=float, default=1e-5)
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--epochs-stage1", type=int, default=200)
    g.add_argument("--epochs-stage2", type=int, default=30)
    g.add_argument("--lambda-1", type=float, default=1.0)
    g.add_argument("--lambda-2", type=float, default=1.0)
    g.add_argument("--lambda-3", type=float, default=1e-3)
    g.add_argument("--lambda-infeasibility", type=float, default=1.0)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--early-stop-patience", type=int, default=25)
    g.add_argument("--scheduler-patience", type=int, default=10)
    g.add_argument("--scheduler-factor", type=float, default=0.5)
    g.add_argument("--num-workers", type=int, default=0)

    # ── thresholds ──
    g = p.add_argument_group("Thresholds")
    g.add_argument("--tau", type=float, default=50.0)
    g.add_argument("--stagnation-tol", type=float, default=1e-6)
    g.add_argument("--vm-min", type=float, default=0.5)
    g.add_argument("--vm-max", type=float, default=1.5)

    # ── misc ──
    g = p.add_argument_group("Misc")
    g.add_argument(
        "--unidirectional-edges",
        action="store_true",
        help="Set if each branch is stored as ONE directed edge.",
    )
    g.add_argument("--amp", action="store_true")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--cpu", action="store_true")

    # ── run mode ──
    g = p.add_argument_group("Run mode")
    g.add_argument(
        "--stage", choices=["1", "2", "both"], default="both"
    )
    g.add_argument("--resume-from", default=None)
    g.add_argument("--eval-only", default=None)

    # ── grid search ──
    g = p.add_argument_group("Grid search")
    g.add_argument(
        "--grid-search",
        action="store_true",
        help="Enable grid-search mode.",
    )
    g.add_argument(
        "--sweep",
        action="append",
        nargs="+",
        metavar=("PARAM", "VAL"),
        help=(
            "Sweep a parameter.  Repeat for each axis.\n"
            "  e.g.  --sweep lambda_1 0.1 1.0 10.0 --sweep lr 1e-4 1e-3\n"
            "If --grid-search is set but no --sweep given, uses defaults."
        ),
    )
    g.add_argument(
        "--grid-config",
        default=None,
        help="JSON file defining the sweep grid (overrides --sweep).",
    )
    g.add_argument(
        "--grid-list",
        action="store_true",
        help="Print all experiments and exit (no training).",
    )
    g.add_argument(
        "--grid-id",
        type=int,
        nargs="+",
        default=None,
        help="Run only these experiment IDs.",
    )
    g.add_argument(
        "--grid-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Run experiments in [START, END).",
    )
    g.add_argument(
        "--grid-results-dir",
        default="grid_results",
        help="Directory for grid-search outputs.",
    )
    return p


def _parse_sweep_spec(args) -> Dict[str, List[Any]]:
    """Build the sweep dict from CLI flags / JSON file / defaults."""
    if args.grid_config:
        with open(args.grid_config) as f:
            raw = json.load(f)
        return {k: [_coerce_value(k, str(v)) for v in vs] for k, vs in raw.items()}

    if args.sweep:
        spec: Dict[str, List[Any]] = {}
        for parts in args.sweep:
            name = parts[0]
            vals = [_coerce_value(name, v) for v in parts[1:]]
            spec[name] = vals
        return spec

    return {k: list(v) for k, v in DEFAULT_SWEEP_GRID.items()}


def _cfg_from_args(args) -> Config:
    return Config(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        wandb_project=args.wandb_project,
        hidden_dim=args.hidden_dim,
        num_mp_layers=args.num_mp_layers,
        dropout=args.dropout,
        use_global_attention=not args.no_global_attention,
        num_attention_heads=args.num_attention_heads,
        rank_k=args.rank_k,
        T=args.T,
        epsilon=args.epsilon,
        max_iter_inference=args.max_iter_inference,
        use_adaptive_lm=args.adaptive_lm,
        mu_init=args.mu_init,
        mu_min=args.mu_min,
        mu_max=args.mu_max,
        mu_decrease=args.mu_decrease,
        mu_increase=args.mu_increase,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        lambda_3=args.lambda_3,
        lambda_infeasibility=args.lambda_infeasibility,
        grad_clip=args.grad_clip,
        early_stop_patience=args.early_stop_patience,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        tau=args.tau,
        stagnation_tol=args.stagnation_tol,
        vm_min=args.vm_min,
        vm_max=args.vm_max,
        bidirectional_edges=not args.unidirectional_edges,
        use_amp=args.amp,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = _cfg_from_args(args)

    seed_everything(cfg.seed)
    device = (
        torch.device("cpu")
        if (args.cpu or not torch.cuda.is_available())
        else torch.device("cuda")
    )
    log.info("Device: %s", device)

    # ── grid search mode ──
    if args.grid_search:
        sweep_spec = _parse_sweep_spec(args)
        experiments = build_experiment_grid(sweep_spec)
        log.info(
            "Grid search: %d experiments  (%d axes: %s)",
            len(experiments),
            len(sweep_spec),
            ", ".join(sweep_spec.keys()),
        )

        if args.grid_list:
            for i, exp in enumerate(experiments):
                print(f"  [{i:4d}]  {exp}")
            print(f"\nTotal: {len(experiments)} experiments")
            return

        exp_ids: Optional[List[int]] = None
        if args.grid_id is not None:
            exp_ids = args.grid_id
        elif args.grid_range is not None:
            exp_ids = list(range(args.grid_range[0], args.grid_range[1]))
        # else: None → run all

        runner = GridSearchRunner(
            base_cfg=cfg,
            experiments=experiments,
            device=device,
            results_dir=args.grid_results_dir,
            stages=args.stage,
        )
        runner.run(experiment_ids=exp_ids)
        if _WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
        return

    # ── single-run mode ──
    train_loader, val_loader, test_loader, norm_stats = load_datasets(cfg)
    log.info(
        "Data — train: %d  val: %d  test: %d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    model = BifurcationAwarePFSolver(cfg, norm_stats)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "Parameters: %d  |  adaptive_lm=%s  global_attn=%s  bidir_edges=%s",
        n_params,
        cfg.use_adaptive_lm,
        cfg.use_global_attention,
        cfg.bidirectional_edges,
    )

    def _load_ckpt(path: str):
        state = torch.load(path, map_location="cpu", weights_only=False)
        sd = state.get("model_state_dict", state)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            log.info("Missing keys: %s", missing)
        if unexpected:
            log.info("Unexpected keys: %s", unexpected)
        log.info("Loaded checkpoint ← %s", path)

    if args.eval_only:
        run_sig = generate_run_signature(cfg)
        setup_file_logging(cfg.log_dir, prefix=f"eval_{run_sig}")
        _load_ckpt(args.eval_only)
        model = model.to(device)
        if _WANDB_AVAILABLE:
            wandb.init(
                project=cfg.wandb_project, config=asdict(cfg), reinit=True
            )
        engine = InferenceEngine(cfg, device)
        engine.run(model, test_loader)
        if _WANDB_AVAILABLE:
            wandb.finish()
        return

    if args.resume_from:
        _load_ckpt(args.resume_from)
    elif args.stage == "2":
        log.warning(
            "Stage 2 without --resume-from: starting from random init."
        )

    model = model.to(device)
    trainer = Trainer(model, train_loader, val_loader, cfg, device)
    trainer.train(stages=args.stage)

    log.info("Running test inference …")
    engine = InferenceEngine(cfg, device)
    engine.run(model, test_loader)

    if _WANDB_AVAILABLE:
        wandb.finish()
    log.info("Done.")


if __name__ == "__main__":
    main()