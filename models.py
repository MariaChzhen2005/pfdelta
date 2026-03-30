"""
Implementation of:
  1. Shared GNN encoder (GINEConv message-passing)
  2. State predictor head  (voltage magnitudes & angles)
  3. Bifurcation-aware regularizer head  (Λ = D + U S Uᵀ)
  4. Differentiable unrolled Newton-Raphson optimisation layer
  5. Two-stage curriculum training with wandb tracking
  6. Inference pipeline with infeasibility detection

Designed to consume the PyG Data objects produced by data_generation.py.
"""

import os
import time
import logging
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.data import Data, Batch
from tqdm import tqdm

import wandb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def setup_file_logging(log_dir: str) -> str:
    """Attach a file handler to the root logger, writing to ``log_dir/run.log``.

    Returns the path to the log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run.log")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)
    log.info("Console output also saved → %s", log_path)
    return log_path

@dataclass
class Config:
    # GNN encoder
    hidden_dim: int = 128
    num_mp_layers: int = 4
    dropout: float = 0.1

    # Regulariser head
    rank_k: int = 8

    # Unrolled solver
    T: int = 5
    epsilon: float = 1e-7
    max_iter_inference: int = 20
    use_implicit_diff: bool = False
    lsqr_damp: float = 0.0

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs_stage1: int = 200
    epochs_stage2: int = 100
    lambda_1: float = 1.0
    lambda_2: float = 1.0
    lambda_3: float = 1e-3
    grad_clip: float = 1.0

    # LR scheduler
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5

    # Mixed precision
    use_amp: bool = False

    # Infeasibility detection
    tau: float = 50.0
    stagnation_tol: float = 1e-6

    # Paths
    data_dir: str = "data/processed/task4_solvability"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    wandb_project: str = "pfdelta-bifurcation"

    # Reproducibility
    seed: int = 42


def load_datasets(cfg: Config):
    """Load train/val/test .pt lists and normalisation statistics."""
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

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, norm_stats


def denormalize_node_features(
    x_norm: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Recover raw physical node quantities from normalised tensor.

    x_norm : [N, 7]  normalised (pd, qd, pg, vm, gs, bs, bus_type)
    Returns dict with raw tensors on the same device.
    """
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
    ea_norm: torch.Tensor,
    ea_mean: torch.Tensor,
    ea_std: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Recover raw branch parameters from normalised edge_attr.

    ea_norm : [2E, 8]  normalised (r, x, g_fr, b_fr, g_to, b_to, tap, shift)
    """
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


class PowerFlowPhysics:
    """Differentiable per-graph power-flow mismatch and Jacobian.

    All methods are static and operate on single-graph tensors so that the
    unrolled solver can call them inside a per-graph loop.
    """

    @staticmethod
    def compute_power_injections(
        va: torch.Tensor,
        vm: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr_raw: Dict[str, torch.Tensor],
        gs: torch.Tensor,
        bs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute bus-level P and Q injections from the current state.

        Uses the standard pi-model branch flow formulation (same physics as
        pf_losses_utils.py).  All ops are pure-PyTorch for autograd support.

        Returns (P_calc, Q_calc) each of shape [N].
        """
        src, dst = edge_index
        n = va.shape[0]

        r = edge_attr_raw["br_r"]
        x = edge_attr_raw["br_x"]
        b_line = edge_attr_raw["b_fr"] + edge_attr_raw["b_to"]
        tau = edge_attr_raw["tap"]
        shift = edge_attr_raw["shift"]

        y_complex = 1.0 / torch.complex(r, x)
        Y_real = y_complex.real
        Y_imag = y_complex.imag

        v_i, v_j = vm[src], vm[dst]
        th_i, th_j = va[src], va[dst]
        d_theta_fwd = th_i - th_j
        d_theta_rev = th_j - th_i

        P_flow_src = (
            v_i * v_j / tau * (
                -Y_real * torch.cos(d_theta_fwd - shift)
                - Y_imag * torch.sin(d_theta_fwd - shift)
            )
            + Y_real * (v_i / tau) ** 2
        )
        P_flow_dst = (
            v_j * v_i / tau * (
                -Y_real * torch.cos(d_theta_rev - shift)
                - Y_imag * torch.sin(d_theta_rev - shift)
            )
            + Y_real * v_j ** 2
        )

        Q_flow_src = (
            v_i * v_j / tau * (
                -Y_real * torch.sin(d_theta_fwd - shift)
                + Y_imag * torch.cos(d_theta_fwd - shift)
            )
            - (Y_imag + b_line / 2) * (v_i / tau) ** 2
        )
        Q_flow_dst = (
            v_j * v_i / tau * (
                -Y_real * torch.sin(d_theta_rev - shift)
                + Y_imag * torch.cos(d_theta_rev - shift)
            )
            - (Y_imag + b_line / 2) * v_j ** 2
        )

        P_calc = torch.zeros(n, device=va.device, dtype=va.dtype)
        P_calc = P_calc.scatter_add(0, src, P_flow_src).scatter_add(0, dst, P_flow_dst)
        P_calc = P_calc + vm ** 2 * gs

        Q_calc = torch.zeros(n, device=va.device, dtype=va.dtype)
        Q_calc = Q_calc.scatter_add(0, src, Q_flow_src).scatter_add(0, dst, Q_flow_dst)
        Q_calc = Q_calc - vm ** 2 * bs

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
        """Compute the 2N-dimensional mismatch vector F(x).

        F is assembled per bus type:
          PQ  (type 1) : F[i] = P_spec - P_calc,   F[N+i] = Q_spec - Q_calc
          PV  (type 2) : F[i] = P_spec - P_calc,   F[N+i] = vm - vm_setpoint
          Slack (type 3): F[i] = va - 0,            F[N+i] = vm - vm_setpoint

        Returns F of shape [2N].
        """
        n = va.shape[0]
        P_calc, Q_calc = PowerFlowPhysics.compute_power_injections(
            va, vm, edge_index, edge_attr_raw, gs, bs,
        )

        F = torch.zeros(2 * n, device=va.device, dtype=va.dtype)

        pq_mask = bus_type == 1
        pv_mask = bus_type == 2
        sl_mask = bus_type == 3

        # P-equations (first N entries)
        F[:n] = p_spec - P_calc
        # Override slack P-equation with angle-fixing constraint
        if sl_mask.any():
            F[:n][sl_mask] = va[sl_mask] - 0.0

        # Q-equations (last N entries)
        F[n:] = q_spec - Q_calc
        # Override PV and slack Q-equations with voltage-fixing constraint
        if pv_mask.any():
            F[n:][pv_mask] = vm[pv_mask] - vm_setpoint[pv_mask]
        if sl_mask.any():
            F[n:][sl_mask] = vm[sl_mask] - vm_setpoint[sl_mask]

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
        """Wrapper that takes a flat x = [va; vm] and returns F(x)."""
        n = x.shape[0] // 2
        va = x[:n]
        vm = x[n:]
        return PowerFlowPhysics.compute_mismatch(
            va, vm, edge_index, edge_attr_raw,
            p_spec, q_spec, gs, bs, bus_type, vm_setpoint,
        )

    @staticmethod
    def compute_jacobian(
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
        """Compute J = dF/dx [2N x 2N] analytically via closed-form pi-model derivatives.

        Much faster than ``torch.autograd.functional.jacobian`` which requires
        2N forward passes through the mismatch function.  The analytical version
        computes all partial derivatives in a single vectorised pass over edges.
        """
        n = x.shape[0] // 2
        va = x[:n]
        vm = x[n:]
        device = x.device
        dtype = x.dtype

        src, dst = edge_index

        r = edge_attr_raw["br_r"]
        xr = edge_attr_raw["br_x"]
        b_line = edge_attr_raw["b_fr"] + edge_attr_raw["b_to"]
        tau = edge_attr_raw["tap"]
        shift = edge_attr_raw["shift"]

        y_complex = 1.0 / torch.complex(r, xr)
        Y_r = y_complex.real
        Y_x = y_complex.imag

        v_a, v_b = vm[src], vm[dst]
        th_a, th_b = va[src], va[dst]
        dtf = th_a - th_b
        dtr = -dtf

        cos_f = torch.cos(dtf - shift)
        sin_f = torch.sin(dtf - shift)
        cos_r = torch.cos(dtr - shift)
        sin_r = torch.sin(dtr - shift)

        vab_tau = v_a * v_b / tau

        # -- Per-edge derivatives of P_flow_src (contributes to P_calc[src]) --
        dPfs_dth_a = vab_tau * (Y_r * sin_f - Y_x * cos_f)
        dPfs_dth_b = -dPfs_dth_a
        dPfs_dv_a = (v_b / tau * (-Y_r * cos_f - Y_x * sin_f)
                     + 2 * Y_r * v_a / tau ** 2)
        dPfs_dv_b = v_a / tau * (-Y_r * cos_f - Y_x * sin_f)

        # -- Per-edge derivatives of P_flow_dst (contributes to P_calc[dst]) --
        dPfd_dth_b = vab_tau * (Y_r * sin_r - Y_x * cos_r)
        dPfd_dth_a = -dPfd_dth_b
        dPfd_dv_b = (v_a / tau * (-Y_r * cos_r - Y_x * sin_r)
                     + 2 * Y_r * v_b)
        dPfd_dv_a = v_b / tau * (-Y_r * cos_r - Y_x * sin_r)

        # -- Per-edge derivatives of Q_flow_src (contributes to Q_calc[src]) --
        dQfs_dth_a = vab_tau * (-Y_r * cos_f - Y_x * sin_f)
        dQfs_dth_b = -dQfs_dth_a
        dQfs_dv_a = (v_b / tau * (-Y_r * sin_f + Y_x * cos_f)
                     - 2 * (Y_x + b_line / 2) * v_a / tau ** 2)
        dQfs_dv_b = v_a / tau * (-Y_r * sin_f + Y_x * cos_f)

        # -- Per-edge derivatives of Q_flow_dst (contributes to Q_calc[dst]) --
        dQfd_dth_b = vab_tau * (-Y_r * cos_r - Y_x * sin_r)
        dQfd_dth_a = -dQfd_dth_b
        dQfd_dv_b = (v_a / tau * (-Y_r * sin_r + Y_x * cos_r)
                     - 2 * (Y_x + b_line / 2) * v_b)
        dQfd_dv_a = v_b / tau * (-Y_r * sin_r + Y_x * cos_r)

        # -- Assemble four N×N sub-blocks via scatter_add --
        # F = [p_spec - P_calc ; q_spec - Q_calc], so dF/dx = -dCalc/dx
        z = torch.zeros(n * n, device=device, dtype=dtype)

        # Block (0,0): dF_P / d_va  =  -dP_calc / d_va
        b00 = z.scatter_add(0, src * n + src, -dPfs_dth_a)
        b00 = b00.scatter_add(0, src * n + dst, -dPfs_dth_b)
        b00 = b00.scatter_add(0, dst * n + dst, -dPfd_dth_b)
        b00 = b00.scatter_add(0, dst * n + src, -dPfd_dth_a)
        dFP_dva = b00.view(n, n)

        # Block (0,1): dF_P / d_vm  =  -dP_calc / d_vm
        b01 = z.scatter_add(0, src * n + src, -dPfs_dv_a)
        b01 = b01.scatter_add(0, src * n + dst, -dPfs_dv_b)
        b01 = b01.scatter_add(0, dst * n + dst, -dPfd_dv_b)
        b01 = b01.scatter_add(0, dst * n + src, -dPfd_dv_a)
        dFP_dvm = b01.view(n, n) - torch.diag(2 * vm * gs)

        # Block (1,0): dF_Q / d_va  =  -dQ_calc / d_va
        b10 = z.scatter_add(0, src * n + src, -dQfs_dth_a)
        b10 = b10.scatter_add(0, src * n + dst, -dQfs_dth_b)
        b10 = b10.scatter_add(0, dst * n + dst, -dQfd_dth_b)
        b10 = b10.scatter_add(0, dst * n + src, -dQfd_dth_a)
        dFQ_dva = b10.view(n, n)

        # Block (1,1): dF_Q / d_vm  =  -dQ_calc / d_vm
        # Shunt term:  d(-Q_calc)/d_vm  for shunt is +2*vm*bs
        b11 = z.scatter_add(0, src * n + src, -dQfs_dv_a)
        b11 = b11.scatter_add(0, src * n + dst, -dQfs_dv_b)
        b11 = b11.scatter_add(0, dst * n + dst, -dQfd_dv_b)
        b11 = b11.scatter_add(0, dst * n + src, -dQfd_dv_a)
        dFQ_dvm = b11.view(n, n) + torch.diag(2 * vm * bs)

        J = torch.cat([
            torch.cat([dFP_dva, dFP_dvm], dim=1),
            torch.cat([dFQ_dva, dFQ_dvm], dim=1),
        ], dim=0)

        # -- Bus-type row overrides (non-in-place for autograd safety) --
        sl_mask = bus_type == 3
        pv_mask = bus_type == 2

        keep_p = (~sl_mask).float()
        keep_q = (~(pv_mask | sl_mask)).float()
        keep = torch.cat([keep_p, keep_q]).unsqueeze(1)
        J = J * keep

        override_diag = torch.zeros(2 * n, device=device, dtype=dtype)
        override_diag[:n] = sl_mask.float()
        override_diag[n:] = (pv_mask | sl_mask).float()
        J = J + torch.diag(override_diag)

        return J

class SharedEncoder(nn.Module):
    """Message-passing encoder producing per-node embeddings H ∈ R^{N×d}.

    Uses GINEConv layers (Graph Isomorphism Network with Edge features)
    with residual connections and layer normalisation.
    """

    def __init__(self, node_dim: int = 7, edge_dim: int = 8,
                 hidden_dim: int = 128, num_layers: int = 4,
                 dropout: float = 0.1):
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

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.node_proj(x)
        e = self.edge_proj(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, e)
            h_new = norm(h_new)
            h_new = F_func.leaky_relu(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new  # residual
        return h



class StatePredictorHead(nn.Module):
    """Maps node embeddings H → x_pred ∈ R^{2N}.

    Predicts (Δva, Δvm) per node, added to a flat-start initialisation:
      PQ  : va=0, vm=1
      PV  : va=0, vm=vm_setpoint
      Slack: va=0, vm=vm_setpoint
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, H: torch.Tensor, bus_type: torch.Tensor,
                vm_setpoint: torch.Tensor) -> torch.Tensor:
        """Return x_pred [2N] = [va_0..va_{N-1}, vm_0..vm_{N-1}]."""
        delta = self.mlp(H)  # [N, 2]  (Δva, Δvm)
        n = H.shape[0]

        va_init = torch.zeros(n, device=H.device, dtype=H.dtype)
        vm_init = torch.ones(n, device=H.device, dtype=H.dtype)

        pv_or_slack = (bus_type == 2) | (bus_type == 3)
        vm_init[pv_or_slack] = vm_setpoint[pv_or_slack]

        va_pred = va_init + delta[:, 0]
        vm_pred = vm_init + delta[:, 1]

        return torch.cat([va_pred, vm_pred], dim=0)

class RegularizerHead(nn.Module):
    """Maps node embeddings H → (D, U, S) defining Λ = D + U S Uᵀ.

    D ∈ R^{2N}  non-negative diagonal (softplus)
    U ∈ R^{2N×k}  low-rank coupling basis
    S ∈ R^{k}     positive diagonal weights (softplus)
    """

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

    def forward(self, H: torch.Tensor, batch: torch.Tensor):
        """Return per-graph (D, U, S) packed into a list of tuples.

        Because graphs in a batch may have different sizes, outputs are
        returned as lists indexed by graph id.  Each element is a tuple
        (d [2n], U [2n, k], s [k]) for that graph.
        """
        d_raw = self.d_mlp(H)       # [N_total, 2]
        u_raw = self.u_mlp(H)       # [N_total, 2k]
        h_pool = global_mean_pool(H, batch)  # [B, hidden_dim]
        s_raw = self.s_mlp(h_pool)   # [B, k]

        d_all = F_func.softplus(d_raw)       # [N_total, 2]
        s_all = F_func.softplus(s_raw)       # [B, k]

        unique_graphs = batch.unique()
        results = []
        for g in unique_graphs:
            mask = batch == g
            n_g = mask.sum().item()
            d_g = d_all[mask].reshape(2 * n_g)           # [2n]
            u_g = u_raw[mask].reshape(2 * n_g, self.rank_k)  # [2n, k]
            s_g = s_all[g]                                    # [k]
            results.append((d_g, u_g, s_g))

        return results
    
def _solve_regularised_step(
    J: torch.Tensor,
    F_val: torch.Tensor,
    d: torch.Tensor,
    U: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    """Solve the regularised Newton sub-problem for Δx.

    Solves the normal equations of min ||A Δx - b||²:

        (Jᵀ J + Λ) Δx = -Jᵀ F,     Λ = diag(d) + U diag(s) Uᵀ

    Uses a single [2N, 2N] linear solve instead of constructing the
    tall stacked matrix and calling lstsq.

    Parameters
    ----------
    J : [2N, 2N]   Jacobian
    F_val : [2N]   mismatch vector F(x_t)
    d : [2N]       diagonal damping  (non-negative)
    U : [2N, k]    low-rank basis
    s : [k]        positive diagonal weights

    Returns
    -------
    dx : [2N]
    """
    JtJ = J.T @ J
    Lambda = torch.diag(d) + U @ torch.diag(s) @ U.T
    H = JtJ + Lambda
    rhs = -(J.T @ F_val)
    dx = torch.linalg.solve(H, rhs.unsqueeze(-1)).squeeze(-1)
    return dx


class ImplicitSolverFn(torch.autograd.Function):
    """Custom autograd function implementing implicit differentiation.

    Forward runs the Newton loop *without* building the unrolled graph.
    Backward applies the implicit function theorem at the converged point.
    """

    @staticmethod
    def forward(
        ctx,
        x_pred: torch.Tensor,
        d: torch.Tensor,
        U: torch.Tensor,
        s: torch.Tensor,
        T: int,
        edge_index: torch.Tensor,
        edge_attr_raw_flat: torch.Tensor,
        p_spec: torch.Tensor,
        q_spec: torch.Tensor,
        gs: torch.Tensor,
        bs: torch.Tensor,
        bus_type: torch.Tensor,
        vm_setpoint: torch.Tensor,
        edge_attr_keys: list,
    ):
        n = x_pred.shape[0] // 2
        ea_raw = _unflatten_edge_attr(edge_attr_raw_flat, edge_attr_keys)

        x = x_pred.detach().clone()
        residual_norms = []

        epsilon = 1e-7
        for _ in range(T):
            with torch.no_grad():
                F_val = PowerFlowPhysics.compute_mismatch_from_x(
                    x, edge_index, ea_raw,
                    p_spec, q_spec, gs, bs, bus_type, vm_setpoint,
                )
                res_norm = F_val.abs().max().item()
                residual_norms.append(res_norm)
                if res_norm < epsilon:
                    break
                J = PowerFlowPhysics.compute_jacobian(
                    x, edge_index, ea_raw,
                    p_spec, q_spec, gs, bs, bus_type, vm_setpoint,
                )
                dx = _solve_regularised_step(J, F_val, d.detach(), U.detach(), s.detach())
                x = x + dx

        ctx.save_for_backward(x, d, U, s, edge_index, edge_attr_raw_flat,
                              p_spec, q_spec, gs, bs, bus_type, vm_setpoint,
                              x_pred)
        ctx.edge_attr_keys = edge_attr_keys
        ctx.residual_norms = residual_norms
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (x_star, d, U, s, edge_index, ea_flat,
         p_spec, q_spec, gs, bs, bus_type, vm_setpoint,
         x_pred) = ctx.saved_tensors
        ea_raw = _unflatten_edge_attr(ea_flat, ctx.edge_attr_keys)

        J = PowerFlowPhysics.compute_jacobian(
            x_star, edge_index, ea_raw,
            p_spec, q_spec, gs, bs, bus_type, vm_setpoint,
        )

        Lambda = torch.diag(d) + U @ torch.diag(s) @ U.T
        H = J.T @ J + Lambda
        v = torch.linalg.solve(H, grad_output.unsqueeze(-1)).squeeze(-1)

        grad_x_pred = Lambda @ v
        diff = (x_star - x_pred).detach()
        grad_d = v * diff
        grad_U = torch.outer(diff, v) @ U @ torch.diag(s) + \
                 torch.outer(v, diff) @ U @ torch.diag(s)
        grad_U = grad_U.T.T  # keep shape [2N, k]
        grad_s = (U.T @ torch.outer(v, diff) @ U).diag()

        return (grad_x_pred, grad_d, grad_U, grad_s,
                None, None, None, None, None, None, None, None, None, None)


def _flatten_edge_attr(ea_raw: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, list]:
    """Pack edge-attr dict into a single tensor for save_for_backward."""
    keys = sorted(ea_raw.keys())
    tensors = [ea_raw[k] for k in keys]
    return torch.stack(tensors, dim=-1), keys


def _unflatten_edge_attr(flat: torch.Tensor, keys: list) -> Dict[str, torch.Tensor]:
    """Unpack the flat tensor back to a dict."""
    return {k: flat[:, i] for i, k in enumerate(keys)}


class UnrolledSolver(nn.Module):
    """Differentiable unrolled Newton-Raphson with regularisation.

    Supports two execution paths:
      - **Batched** (default): groups same-size graphs, stacks tensors, and
        runs a single ``torch.linalg.solve`` per solver step.
      - **Implicit differentiation**: per-graph ``ImplicitSolverFn`` (custom
        autograd backward via the implicit function theorem).
    """

    def __init__(self, T: int = 5, use_implicit_diff: bool = False,
                 epsilon: float = 1e-7):
        super().__init__()
        self.T = T
        self.use_implicit_diff = use_implicit_diff
        self.epsilon = epsilon

    # ------------------------------------------------------------------ #
    #  Single-graph path (used by ImplicitSolverFn)                       #
    # ------------------------------------------------------------------ #

    def forward_single_graph(
        self,
        x_pred: torch.Tensor,
        d: torch.Tensor,
        U: torch.Tensor,
        s: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr_raw: Dict[str, torch.Tensor],
        p_spec: torch.Tensor,
        q_spec: torch.Tensor,
        gs: torch.Tensor,
        bs_node: torch.Tensor,
        bus_type: torch.Tensor,
        vm_setpoint: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[float]]:
        """Run the unrolled solver on a single graph."""
        ea_flat, ea_keys = _flatten_edge_attr(edge_attr_raw)
        x_final = ImplicitSolverFn.apply(
            x_pred, d, U, s, self.T,
            edge_index, ea_flat,
            p_spec, q_spec, gs, bs_node, bus_type, vm_setpoint,
            ea_keys,
        )
        return x_final, []

    def forward_batch(
        self,
        x_pred_list: List[torch.Tensor],
        reg_params: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        per_graph: List[Dict[str, torch.Tensor]],
    ) -> Tuple[List[torch.Tensor], List[List[float]]]:
        """Run the solver on a batch of graphs with batched linear solves.

        Groups graphs by node count so that ``torch.linalg.solve`` processes
        all same-size graphs in a single kernel call per solver step.

        Returns (x_final_list, residuals_list) in the original graph order.
        """
        B = len(x_pred_list)
        device = x_pred_list[0].device
        dtype = x_pred_list[0].dtype

        groups: Dict[int, List[int]] = {}
        for i, gi in enumerate(per_graph):
            groups.setdefault(gi["n"], []).append(i)

        x_final_out: List[torch.Tensor] = [torch.empty(0)] * B
        residuals_out: List[List[float]] = [[] for _ in range(B)]

        for n_nodes, indices in groups.items():
            Bg = len(indices)
            dim = 2 * n_nodes

            x = torch.stack([x_pred_list[i] for i in indices])        # [Bg, 2N]
            d_b = torch.stack([reg_params[i][0] for i in indices])    # [Bg, 2N]
            U_b = torch.stack([reg_params[i][1] for i in indices])    # [Bg, 2N, k]
            s_b = torch.stack([reg_params[i][2] for i in indices])    # [Bg, k]

            Lambda = (torch.diag_embed(d_b)
                      + U_b @ torch.diag_embed(s_b) @ U_b.transpose(1, 2))

            g_infos = [per_graph[i] for i in indices]

            for _t in range(self.T):
                F_list = []
                J_list = []
                for b in range(Bg):
                    gi = g_infos[b]
                    F_val = PowerFlowPhysics.compute_mismatch_from_x(
                        x[b], gi["edge_index"], gi["edge_attr_raw"],
                        gi["p_spec"], gi["q_spec"],
                        gi["gs"], gi["bs"], gi["bus_type"], gi["vm_setpoint"],
                    )
                    J_val = PowerFlowPhysics.compute_jacobian(
                        x[b], gi["edge_index"], gi["edge_attr_raw"],
                        gi["p_spec"], gi["q_spec"],
                        gi["gs"], gi["bs"], gi["bus_type"], gi["vm_setpoint"],
                    )
                    F_list.append(F_val)
                    J_list.append(J_val)

                F_batch = torch.stack(F_list)              # [Bg, 2N]

                max_res = F_batch.detach().abs().amax(dim=1).max().item()
                if max_res < self.epsilon:
                    break

                J_batch = torch.stack(J_list)              # [Bg, 2N, 2N]

                JtJ = J_batch.transpose(1, 2) @ J_batch   # [Bg, 2N, 2N]
                H = JtJ + Lambda                           # [Bg, 2N, 2N]
                rhs = -(J_batch.transpose(1, 2)
                        @ F_batch.unsqueeze(-1)).squeeze(-1)  # [Bg, 2N]

                dx = torch.linalg.solve(H, rhs.unsqueeze(-1)).squeeze(-1)
                x = x + dx

            for b, idx in enumerate(indices):
                x_final_out[idx] = x[b]

        return x_final_out, residuals_out


class BifurcationAwarePFSolver(nn.Module):
    """Complete model combining encoder, dual heads, and unrolled solver.

    Registers normalisation stats as buffers so they travel with the model.
    """

    def __init__(self, cfg: Config, norm_stats: Dict[str, torch.Tensor]):
        super().__init__()
        self.cfg = cfg

        self.encoder = SharedEncoder(
            node_dim=7, edge_dim=8,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_mp_layers,
            dropout=cfg.dropout,
        )
        self.state_head = StatePredictorHead(hidden_dim=cfg.hidden_dim)
        self.reg_head = RegularizerHead(
            hidden_dim=cfg.hidden_dim, rank_k=cfg.rank_k,
        )
        self.solver = UnrolledSolver(
            T=cfg.T, use_implicit_diff=cfg.use_implicit_diff,
            epsilon=cfg.epsilon,
        )

        self.register_buffer("x_mean", norm_stats["x_mean"])
        self.register_buffer("x_std", norm_stats["x_std"])
        self.register_buffer("edge_mean", norm_stats["edge_mean"])
        self.register_buffer("edge_std", norm_stats["edge_std"])

    def _extract_per_graph(
        self, data: Batch, node_raw: Dict[str, torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        """Extract raw physics quantities for each graph in the batch."""
        edge_raw = denormalize_edge_features(
            data.edge_attr, self.edge_mean, self.edge_std,
        )
        batch_idx = data.batch
        edge_src = data.edge_index[0]

        unique_graphs = batch_idx.unique()
        per_graph = []

        for g in unique_graphs:
            node_mask = batch_idx == g
            n_g = node_mask.sum().item()
            node_offset = node_mask.nonzero(as_tuple=True)[0][0].item()

            edge_mask = (edge_src >= node_offset) & (edge_src < node_offset + n_g)
            ei_g = data.edge_index[:, edge_mask] - node_offset

            ea_g = {k: v[edge_mask] for k, v in edge_raw.items()}

            per_graph.append({
                "edge_index": ei_g,
                "edge_attr_raw": ea_g,
                "p_spec": node_raw["pg"][node_mask] - node_raw["pd"][node_mask],
                "q_spec": -node_raw["qd"][node_mask],
                "gs": node_raw["gs"][node_mask],
                "bs": node_raw["bs"][node_mask],
                "bus_type": node_raw["bus_type"][node_mask],
                "vm_setpoint": node_raw["vm_setpoint"][node_mask],
                "n": n_g,
            })

        return per_graph

    def forward(
        self,
        data: Batch,
        run_solver: bool = True,
    ) -> Dict[str, torch.Tensor]:
        H = self.encoder(data.x, data.edge_index, data.edge_attr)

        node_raw = denormalize_node_features(data.x, self.x_mean, self.x_std)

        x_pred = self.state_head(
            H, node_raw["bus_type"], node_raw["vm_setpoint"],
        )

        reg_params = self.reg_head(H, data.batch)

        per_graph = self._extract_per_graph(data, node_raw)

        batch_idx = data.batch
        unique_graphs = batch_idx.unique()
        total_nodes = data.x.shape[0]

        x_pred_list = []
        offset = 0
        for g_info in per_graph:
            n_g = g_info["n"]
            va_g = x_pred[offset:offset + n_g]
            vm_g = x_pred[total_nodes + offset:total_nodes + offset + n_g]
            x_pred_list.append(torch.cat([va_g, vm_g]))
            offset += n_g

        if run_solver:
            if self.solver.use_implicit_diff:
                x_final_list = []
                all_residuals = []
                for i, (g_info, (d_g, u_g, s_g)) in enumerate(
                    zip(per_graph, reg_params)
                ):
                    x_f, res = self.solver.forward_single_graph(
                        x_pred_list[i], d_g, u_g, s_g,
                        g_info["edge_index"], g_info["edge_attr_raw"],
                        g_info["p_spec"], g_info["q_spec"],
                        g_info["gs"], g_info["bs"],
                        g_info["bus_type"], g_info["vm_setpoint"],
                    )
                    x_final_list.append(x_f)
                    all_residuals.append(res)
            else:
                x_final_list, all_residuals = self.solver.forward_batch(
                    x_pred_list, reg_params, per_graph,
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
            "residuals": all_residuals,
        }


def loss_state(
    x_pred: torch.Tensor,
    y_state: torch.Tensor,
    feasible_mask: torch.Tensor,
    batch: torch.Tensor,
    total_nodes: int,
) -> torch.Tensor:
    """L_state = ||x_pred - x*||²  (only on feasible samples).

    x_pred : [2*total_nodes]  packed as [va_all; vm_all]
    y_state : [total_nodes, 2]  columns are [va, vm]
    feasible_mask : [B]  bool per graph
    """
    va_pred = x_pred[:total_nodes]
    vm_pred = x_pred[total_nodes:]
    va_true = y_state[:, 0]
    vm_true = y_state[:, 1]

    unique_graphs = batch.unique()
    total_loss = torch.tensor(0.0, device=x_pred.device)
    count = 0

    offset = 0
    for i, g in enumerate(unique_graphs):
        n_g = (batch == g).sum().item()
        if feasible_mask[i]:
            total_loss = total_loss + (
                (va_pred[offset:offset + n_g] - va_true[offset:offset + n_g]).pow(2).sum()
                + (vm_pred[offset:offset + n_g] - vm_true[offset:offset + n_g]).pow(2).sum()
            )
            count += 2 * n_g
        offset += n_g

    if count == 0:
        return torch.tensor(0.0, device=x_pred.device, requires_grad=True)
    return total_loss / count


def loss_physics(
    x_final_list: List[torch.Tensor],
    per_graph: List[Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """L_phys = mean over graphs of ||F(x_T)||²."""
    total = torch.tensor(0.0, device=x_final_list[0].device)
    for x_f, g_info in zip(x_final_list, per_graph):
        F_val = PowerFlowPhysics.compute_mismatch_from_x(
            x_f, g_info["edge_index"], g_info["edge_attr_raw"],
            g_info["p_spec"], g_info["q_spec"],
            g_info["gs"], g_info["bs"],
            g_info["bus_type"], g_info["vm_setpoint"],
        )
        total = total + F_val.pow(2).sum()
    return total / len(x_final_list)


def loss_regularisation(
    reg_params: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """L_reg = mean tr(D) + tr(S)  over graphs."""
    total = torch.tensor(0.0, device=reg_params[0][0].device)
    for d_g, u_g, s_g in reg_params:
        total = total + d_g.sum() + s_g.sum()
    return total / len(reg_params)


def composite_loss(
    outputs: Dict,
    data: Batch,
    cfg: Config,
    stage: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the loss for the current training stage.

    Stage 1: L_state only
    Stage 2: λ₁ L_state + λ₂ L_phys + λ₃ L_reg
    """
    total_nodes = data.x.shape[0]
    feasible = data.feasible_mask  # [B]

    l_s = loss_state(
        outputs["x_pred"], data.y_state, feasible,
        data.batch, total_nodes,
    )

    metrics = {"L_state": l_s.item()}

    if stage == 1:
        return l_s, metrics

    l_p = loss_physics(outputs["x_final_list"], outputs["per_graph"])
    l_r = loss_regularisation(outputs["reg_params"])

    total = cfg.lambda_1 * l_s + cfg.lambda_2 * l_p + cfg.lambda_3 * l_r
    metrics.update({
        "L_phys": l_p.item(),
        "L_reg": l_r.item(),
        "L_total": total.item(),
    })
    return total, metrics


# ============================================================================
# SECTION 10: TRAINER
# ============================================================================

class Trainer:
    """Two-stage curriculum trainer with wandb logging and local checkpoints."""

    def __init__(
        self,
        model: BifurcationAwarePFSolver,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Config,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
        )

        self.amp_enabled = cfg.use_amp and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.amp_dtype = torch.float16 if self.amp_enabled else None

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        setup_file_logging(cfg.log_dir)

        wandb.init(
            project=cfg.wandb_project,
            config=asdict(cfg),
            reinit=True,
        )
        wandb.watch(model, log="gradients", log_freq=50)

        self.cumulative_train_time = 0.0

    # --------------------------------------------------------------------- #
    #  Helpers                                                                #
    # --------------------------------------------------------------------- #

    def _save_checkpoint(self, epoch: int, stage: int):
        path = os.path.join(
            self.cfg.checkpoint_dir, f"epoch_{epoch}_stage{stage}.pt",
        )
        torch.save({
            "epoch": epoch,
            "stage": stage,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
        log.info("Checkpoint saved → %s", path)

    @torch.no_grad()
    def _validate(self, stage: int) -> Dict[str, float]:
        self.model.eval()
        running = {}
        count = 0
        val_bar = tqdm(
            self.val_loader,
            desc=f"  val (stage {stage})",
            leave=False,
            unit="batch",
        )
        for data in val_bar:
            data = data.to(self.device)
            outputs = self.model(data, run_solver=(stage == 2))
            _, metrics = composite_loss(outputs, data, self.cfg, stage)
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v
            count += 1
            if "L_state" in running:
                val_bar.set_postfix(
                    L_state=f"{running['L_state'] / count:.4f}",
                )
        return {k: v / max(count, 1) for k, v in running.items()}

    # --------------------------------------------------------------------- #
    #  Stage runners                                                          #
    # --------------------------------------------------------------------- #

    def _run_stage(self, stage: int, num_epochs: int, epoch_offset: int):
        solver_tag = "ON" if stage == 2 else "OFF"
        log.info("=" * 60)
        log.info("STAGE %d  —  %d epochs  (solver %s)", stage, num_epochs, solver_tag)
        log.info("=" * 60)

        epoch_bar = tqdm(
            range(num_epochs),
            desc=f"Stage {stage}",
            unit="epoch",
            position=0,
        )

        for epoch_local in epoch_bar:
            epoch_global = epoch_offset + epoch_local
            self.model.train()
            epoch_loss = 0.0
            epoch_metrics: Dict[str, float] = {}
            n_batches = 0

            t_start = time.time()

            batch_bar = tqdm(
                self.train_loader,
                desc=f"  Ep {epoch_global + 1} train",
                leave=False,
                unit="batch",
                position=1,
            )
            for data in batch_bar:
                data = data.to(self.device)
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    outputs = self.model(data, run_solver=(stage == 2))
                    loss, metrics = composite_loss(
                        outputs, data, self.cfg, stage,
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()
                for k, v in metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
                n_batches += 1

                batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            t_epoch = time.time() - t_start
            self.cumulative_train_time += t_epoch

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_metrics = {k: v / max(n_batches, 1)
                           for k, v in epoch_metrics.items()}

            self.scheduler.step(avg_loss)

            val_metrics = self._validate(stage)

            log_dict = {
                "stage": stage,
                "epoch": epoch_global,
                "train/loss": avg_loss,
                "train/time_s": t_epoch,
                "train/cumulative_time_s": self.cumulative_train_time,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            for k, v in avg_metrics.items():
                log_dict[f"train/{k}"] = v
            for k, v in val_metrics.items():
                log_dict[f"val/{k}"] = v

            wandb.log(log_dict, step=epoch_global)

            summary = (
                f"loss={avg_loss:.5f}  "
                f"val_state={val_metrics.get('L_state', 0.0):.5f}  "
                f"time={t_epoch:.1f}s  "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )
            epoch_bar.set_postfix_str(summary)
            log.info(
                "Epoch %d/%d (stage %d)  %s",
                epoch_global + 1,
                epoch_offset + num_epochs,
                stage,
                summary,
            )

            if (epoch_global + 1) % 5 == 0:
                self._save_checkpoint(epoch_global + 1, stage)

        epoch_bar.close()

    # --------------------------------------------------------------------- #
    #  Main entry                                                             #
    # --------------------------------------------------------------------- #

    def train(self, stages: str = "both"):
        """Run training for the requested stage(s).

        Args:
            stages: ``"1"`` for pre-training only, ``"2"`` for end-to-end
                    unrolling only, or ``"both"`` (default) for the full
                    two-stage curriculum.
        """
        total_start = time.time()
        run_stage1 = stages in ("1", "both")
        run_stage2 = stages in ("2", "both")

        if run_stage1:
            self._run_stage(stage=1, num_epochs=self.cfg.epochs_stage1,
                            epoch_offset=0)
            stage1_path = os.path.join(
                self.cfg.checkpoint_dir, "stage1_final.pt",
            )
            torch.save({
                "stage": 1,
                "epoch": self.cfg.epochs_stage1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            }, stage1_path)
            log.info("Stage 1 model saved → %s", stage1_path)

        if run_stage2:
            epoch_offset = self.cfg.epochs_stage1 if run_stage1 else 0
            self._run_stage(stage=2, num_epochs=self.cfg.epochs_stage2,
                            epoch_offset=epoch_offset)

        total_time = time.time() - total_start
        log.info("Training complete in %.1f s", total_time)
        wandb.log({"total_training_time_s": total_time})

        tag = f"stage{stages}" if stages != "both" else "final"
        final_path = os.path.join(self.cfg.checkpoint_dir, f"{tag}_model.pt")
        torch.save(self.model.state_dict(), final_path)
        log.info("Final model saved → %s", final_path)


# ============================================================================
# SECTION 11: INFERENCE PIPELINE
# ============================================================================

class InferenceEngine:
    """Run inference with convergence checking and infeasibility detection."""

    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device

    @torch.no_grad()
    def run(
        self,
        model: BifurcationAwarePFSolver,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        model.eval()
        orig_T = model.solver.T
        model.solver.T = self.cfg.max_iter_inference

        all_converged: List[bool] = []
        all_pred_infeasible: List[bool] = []
        all_true_feasible: List[bool] = []
        total_samples = 0
        total_inference_time = 0.0

        infer_bar = tqdm(
            dataloader,
            desc="Inference",
            unit="batch",
        )
        for data in infer_bar:
            data = data.to(self.device)
            batch_size_g = data.batch.unique().shape[0]
            total_samples += batch_size_g

            t0 = time.time()
            outputs = model(data, run_solver=True)
            batch_time = time.time() - t0
            total_inference_time += batch_time

            infer_bar.set_postfix(
                samples=total_samples,
                ms_per_sample=f"{1000 * batch_time / max(batch_size_g, 1):.1f}",
            )

            for i, (x_f, g_info, (d_g, u_g, s_g)) in enumerate(
                zip(
                    outputs["x_final_list"],
                    outputs["per_graph"],
                    outputs["reg_params"],
                )
            ):
                F_final = PowerFlowPhysics.compute_mismatch_from_x(
                    x_f,
                    g_info["edge_index"], g_info["edge_attr_raw"],
                    g_info["p_spec"], g_info["q_spec"],
                    g_info["gs"], g_info["bs"],
                    g_info["bus_type"], g_info["vm_setpoint"],
                )
                residual_inf = F_final.abs().max().item()
                converged = residual_inf < self.cfg.epsilon

                trace_lambda = d_g.sum().item() + s_g.sum().item()

                residuals = outputs["residuals"][i]
                stagnated = False
                if len(residuals) >= 2:
                    delta_r = abs(residuals[-1] - residuals[-2])
                    stagnated = delta_r < self.cfg.stagnation_tol

                pred_infeasible = (
                    (not converged)
                    and stagnated
                    and (trace_lambda > self.cfg.tau)
                )

                all_converged.append(converged)
                all_pred_infeasible.append(pred_infeasible)
                all_true_feasible.append(data.feasible_mask[i].item())

        convergence_rate = sum(all_converged) / max(total_samples, 1)

        true_infeasible = [not f for f in all_true_feasible]
        tp = sum(p and t for p, t in zip(all_pred_infeasible, true_infeasible))
        fp = sum(p and (not t) for p, t in zip(all_pred_infeasible, true_infeasible))
        fn = sum((not p) and t for p, t in zip(all_pred_infeasible, true_infeasible))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        per_sample_time = total_inference_time / max(total_samples, 1)

        metrics = {
            "inference/total_time_s": total_inference_time,
            "inference/per_sample_time_s": per_sample_time,
            "inference/convergence_rate": convergence_rate,
            "inference/infeasibility_precision": precision,
            "inference/infeasibility_recall": recall,
            "inference/infeasibility_f1": f1,
            "inference/total_samples": total_samples,
        }

        if wandb.run is not None:
            wandb.log(metrics)
        log.info("Inference complete: %d samples in %.2fs (%.4fs/sample)",
                 total_samples, total_inference_time, per_sample_time)
        log.info("Convergence rate: %.3f", convergence_rate)
        log.info("Infeasibility — P: %.3f  R: %.3f  F1: %.3f",
                 precision, recall, f1)

        model.solver.T = orig_T
        return metrics


# ============================================================================
# SECTION 12: MAIN ENTRY POINT
# ============================================================================

def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train / evaluate the Bifurcation-Aware PF Solver.",
    )
    p.add_argument("--data-dir", default="data/processed/task4_solvability")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--wandb-project", default="pfdelta-bifurcation")

    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-mp-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--rank-k", type=int, default=8)

    p.add_argument("--T", type=int, default=5, help="Unroll steps")
    p.add_argument("--epsilon", type=float, default=1e-7)
    p.add_argument("--max-iter-inference", type=int, default=20)
    p.add_argument("--use-implicit-diff", action="store_true")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs-stage1", type=int, default=50)
    p.add_argument("--epochs-stage2", type=int, default=100)
    p.add_argument("--lambda-1", type=float, default=1.0)
    p.add_argument("--lambda-2", type=float, default=1.0)
    p.add_argument("--lambda-3", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--tau", type=float, default=50.0)
    p.add_argument("--stagnation-tol", type=float, default=1e-6)

    p.add_argument("--amp", action="store_true",
                   help="Enable mixed-precision (AMP) training on CUDA.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--stage", choices=["1", "2", "both"], default="both",
                   help="Which training stage(s) to run: '1', '2', or 'both'.")
    p.add_argument("--resume-from", default=None,
                   help="Path to checkpoint to load before training "
                        "(required when --stage 2 to supply stage-1 weights).")
    p.add_argument("--eval-only", default=None,
                   help="Path to checkpoint for evaluation only.")

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        wandb_project=args.wandb_project,
        hidden_dim=args.hidden_dim,
        num_mp_layers=args.num_mp_layers,
        dropout=args.dropout,
        rank_k=args.rank_k,
        T=args.T,
        epsilon=args.epsilon,
        max_iter_inference=args.max_iter_inference,
        use_implicit_diff=args.use_implicit_diff,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        lambda_3=args.lambda_3,
        grad_clip=args.grad_clip,
        tau=args.tau,
        stagnation_tol=args.stagnation_tol,
        use_amp=args.amp,
        seed=args.seed,
    )

    seed_everything(cfg.seed)

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    log.info("Device: %s", device)

    train_loader, val_loader, test_loader, norm_stats = load_datasets(cfg)
    log.info(
        "Data loaded — train: %d  val: %d  test: %d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    model = BifurcationAwarePFSolver(cfg, norm_stats)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %d", n_params)

    def _load_checkpoint(path: str):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
            log.info("Loaded model weights from checkpoint (dict) → %s", path)
        else:
            model.load_state_dict(state)
            log.info("Loaded model weights from checkpoint (raw) → %s", path)

    if args.eval_only:
        setup_file_logging(cfg.log_dir)
        _load_checkpoint(args.eval_only)
        model = model.to(device)
        wandb.init(project=cfg.wandb_project, config=asdict(cfg), reinit=True)
        engine = InferenceEngine(cfg, device)
        engine.run(model, test_loader)
        wandb.finish()
        return

    if args.resume_from:
        _load_checkpoint(args.resume_from)

    if args.stage == "2" and args.resume_from is None:
        log.warning(
            "Running stage 2 without --resume-from; model starts from "
            "random init (use --resume-from to load stage-1 weights)."
        )

    model = model.to(device)
    trainer = Trainer(model, train_loader, val_loader, cfg, device)
    trainer.train(stages=args.stage)

    log.info("Running inference on test set...")
    engine = InferenceEngine(cfg, device)
    engine.run(model, test_loader)

    wandb.finish()
    log.info("Done.")


if __name__ == "__main__":
    main()
