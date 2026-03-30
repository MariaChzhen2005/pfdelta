import os
import time
import logging
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_func
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


def setup_file_logging(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run.log")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    )
    logging.getLogger().addHandler(fh)
    log.info("Log file → %s", log_path)
    return log_path


# CONFIG
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
    epochs_stage2: int = 100
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

    # Data convention: True iff edge_index stores 2 directed edges per branch
    bidirectional_edges: bool = True

    # Paths
    data_dir: str = "data/processed/task4_solvability"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    wandb_project: str = "pfdelta-bifurcation"
    seed: int = 42


# DATA HELPERS
def load_datasets(cfg: Config):
    train_data = torch.load(os.path.join(cfg.data_dir, "train.pt"), weights_only=False)
    val_data = torch.load(os.path.join(cfg.data_dir, "val.pt"), weights_only=False)
    test_data = torch.load(os.path.join(cfg.data_dir, "test.pt"), weights_only=False)
    norm_stats = torch.load(os.path.join(cfg.data_dir, "norm_stats.pt"), weights_only=False)
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False)
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

# PHYSICS: POWER-FLOW MISMATCH & JACOBIAN
def _scatter_to_block(
    row_idx: torch.Tensor,
    col_idx: torch.Tensor,
    vals: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Build a dense [n, n] matrix from COO entries via scatter-add."""
    flat = row_idx * n + col_idx
    buf = torch.zeros(n * n, device=vals.device, dtype=vals.dtype)
    return buf.scatter_add(0, flat, vals).view(n, n)


class PowerFlowPhysics:
    """Differentiable pi-model power-flow equations.

    All methods are static and operate on single-graph tensors.

    IMPORTANT — edge convention
    ---------------------------
    Each physical branch must appear **once** in `edge_index`.  When the
    dataset stores bidirectional edges (2 directed per branch), filter to
    forward-only *before* calling these methods.  Both the from-side and
    to-side injections are computed for every directed edge that is passed in.
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
        src, dst = edge_index
        n = va.shape[0]

        r = edge_attr_raw["br_r"]
        x = edge_attr_raw["br_x"]
        y_complex = 1.0 / torch.complex(r, x)
        g_s = y_complex.real
        b_s = y_complex.imag

        g_fr = edge_attr_raw["g_fr"]
        b_fr = edge_attr_raw["b_fr"]
        g_to = edge_attr_raw["g_to"]
        b_to = edge_attr_raw["b_to"]
        tau = edge_attr_raw["tap"]
        shift = edge_attr_raw["shift"]

        v_i, v_j = vm[src], vm[dst]
        th_i, th_j = va[src], va[dst]

        # --- trig terms ------------------------------------------------
        # From-side angle: θ_i − θ_j − φ
        cos_f = torch.cos(th_i - th_j - shift)
        sin_f = torch.sin(th_i - th_j - shift)
        # To-side angle: θ_j − θ_i + φ   (FIXED sign on φ)
        cos_t = torch.cos(th_j - th_i + shift)
        sin_t = torch.sin(th_j - th_i + shift)

        vij_tau = v_i * v_j / tau

        # --- from-side (injected at src) --------------------------------
        P_from = (v_i / tau) ** 2 * (g_s + g_fr) + vij_tau * (-g_s * cos_f - b_s * sin_f)
        Q_from = -((v_i / tau) ** 2) * (b_s + b_fr) + vij_tau * (-g_s * sin_f + b_s * cos_f)

        # --- to-side (injected at dst) ----------------------------------
        P_to = v_j ** 2 * (g_s + g_to) + vij_tau * (-g_s * cos_t - b_s * sin_t)
        Q_to = -(v_j ** 2) * (b_s + b_to) + vij_tau * (-g_s * sin_t + b_s * cos_t)

        P_calc = torch.zeros(n, device=va.device, dtype=va.dtype)
        P_calc.scatter_add_(0, src, P_from)
        P_calc.scatter_add_(0, dst, P_to)
        P_calc = P_calc + vm ** 2 * gs

        Q_calc = torch.zeros(n, device=va.device, dtype=va.dtype)
        Q_calc.scatter_add_(0, src, Q_from)
        Q_calc.scatter_add_(0, dst, Q_to)
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
            x[:n], x[n:], edge_index, edge_attr_raw,
            p_spec, q_spec, gs, bs, bus_type, vm_setpoint,
        )

    @staticmethod
    def compute_jacobian(
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr_raw: Dict[str, torch.Tensor],
        gs: torch.Tensor,
        bs: torch.Tensor,
        bus_type: torch.Tensor,
        vm_setpoint: torch.Tensor,
    ) -> torch.Tensor:
        """Analytical 2N×2N Jacobian of the mismatch vector F(x).

        Handles from-side *and* to-side per directed edge, with the
        corrected to-side trig angle (θ_j − θ_i + φ).
        """
        n = x.shape[0] // 2
        va, vm = x[:n], x[n:]
        device, dtype = x.device, x.dtype

        src, dst = edge_index

        r = edge_attr_raw["br_r"]
        x_imp = edge_attr_raw["br_x"]
        y_complex = 1.0 / torch.complex(r, x_imp)
        g_s = y_complex.real
        b_s = y_complex.imag

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
        A_f = g_s * cos_f + b_s * sin_f
        B_f = g_s * sin_f - b_s * cos_f
        A_t = g_s * cos_t + b_s * sin_t
        B_t = g_s * sin_t - b_s * cos_t

        # ---- from-side derivatives (affect row = src) ------------------
        dPfs_dth_src = vij_tau * B_f
        dPfs_dth_dst = -vij_tau * B_f
        dPfs_dvm_src = 2.0 * v_i / tau ** 2 * (g_s + g_fr) - (v_j / tau) * A_f
        dPfs_dvm_dst = -(v_i / tau) * A_f

        dQfs_dth_src = -vij_tau * A_f
        dQfs_dth_dst = vij_tau * A_f
        dQfs_dvm_src = -2.0 * v_i / tau ** 2 * (b_s + b_fr) - (v_j / tau) * B_f
        dQfs_dvm_dst = -(v_i / tau) * B_f

        # ---- to-side derivatives (affect row = dst) --------------------
        dPfd_dth_dst = vij_tau * B_t
        dPfd_dth_src = -vij_tau * B_t
        dPfd_dvm_dst = 2.0 * v_j * (g_s + g_to) - (v_i / tau) * A_t
        dPfd_dvm_src = -(v_j / tau) * A_t

        dQfd_dth_dst = -vij_tau * A_t
        dQfd_dth_src = vij_tau * A_t
        dQfd_dvm_dst = -2.0 * v_j * (b_s + b_to) - (v_i / tau) * B_t
        dQfd_dvm_src = -(v_j / tau) * B_t

        # ---- assemble sub-blocks (F = spec − calc ⇒ dF/d· = −d calc/d·)
        _r = torch.cat  # alias for readability

        dFP_dva = _scatter_to_block(
            _r([src, src, dst, dst]),
            _r([src, dst, dst, src]),
            _r([-dPfs_dth_src, -dPfs_dth_dst, -dPfd_dth_dst, -dPfd_dth_src]),
            n,
        )

        dFP_dvm = (
            _scatter_to_block(
                _r([src, src, dst, dst]),
                _r([src, dst, dst, src]),
                _r([-dPfs_dvm_src, -dPfs_dvm_dst, -dPfd_dvm_dst, -dPfd_dvm_src]),
                n,
            )
            - torch.diag(2.0 * vm * gs)
        )

        dFQ_dva = _scatter_to_block(
            _r([src, src, dst, dst]),
            _r([src, dst, dst, src]),
            _r([-dQfs_dth_src, -dQfs_dth_dst, -dQfd_dth_dst, -dQfd_dth_src]),
            n,
        )

        dFQ_dvm = (
            _scatter_to_block(
                _r([src, src, dst, dst]),
                _r([src, dst, dst, src]),
                _r([-dQfs_dvm_src, -dQfs_dvm_dst, -dQfd_dvm_dst, -dQfd_dvm_src]),
                n,
            )
            + torch.diag(2.0 * vm * bs)
        )

        J = torch.cat(
            [torch.cat([dFP_dva, dFP_dvm], dim=1), torch.cat([dFQ_dva, dFQ_dvm], dim=1)],
            dim=0,
        )

        # ---- bus-type row overrides ------------------------------------
        sl_mask = bus_type == 3
        pv_mask = bus_type == 2

        keep_p = (~sl_mask).float()
        keep_q = (~(pv_mask | sl_mask)).float()
        keep = torch.cat([keep_p, keep_q]).unsqueeze(1)
        J = J * keep

        override = torch.zeros(2 * n, device=device, dtype=dtype)
        override[:n] = sl_mask.float()
        override[n:] = (pv_mask | sl_mask).float()
        J = J + torch.diag(override)
        return J


def _verify_jacobian(x, edge_index, ear, gs, bs, bt, vm_sp, eps=1e-5):
    """Finite-difference Jacobian for debugging (never used in training)."""
    n2 = x.shape[0]
    J_fd = torch.zeros(n2, n2, device=x.device, dtype=x.dtype)
    F0 = PowerFlowPhysics.compute_mismatch_from_x(x, edge_index, ear,
                                                    torch.zeros_like(gs), torch.zeros_like(gs),
                                                    gs, bs, bt, vm_sp)
    for i in range(n2):
        xp = x.clone()
        xp[i] += eps
        Fp = PowerFlowPhysics.compute_mismatch_from_x(xp, edge_index, ear,
                                                        torch.zeros_like(gs), torch.zeros_like(gs),
                                                        gs, bs, bt, vm_sp)
        J_fd[:, i] = (Fp - F0) / eps
    return J_fd


# ============================================================================
# NEURAL NETWORK COMPONENTS
# ============================================================================
class GraphTransformerLayer(nn.Module):
    """Global self-attention over nodes within each graph.

    Pads variable-size graphs to the batch maximum, applies multi-head
    attention with key-padding masks, then unpads.  Complexity is
    O(N_max² · B) per layer.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
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
        d = H.shape[1]
        dev, dt = H.device, H.dtype

        padded = torch.zeros(B, max_n, d, device=dev, dtype=dt)
        key_pad = torch.ones(B, max_n, dtype=torch.bool, device=dev)

        splits = torch.split(H, counts.tolist())
        for i, h in enumerate(splits):
            ni = h.shape[0]
            padded[i, :ni] = h
            key_pad[i, :ni] = False

        attn_out, _ = self.mha(padded, padded, padded, key_padding_mask=key_pad)
        padded = self.norm1(padded + attn_out)
        padded = self.norm2(padded + self.ffn(padded))

        parts = [padded[i, : counts[i]] for i in range(B)]
        return torch.cat(parts, dim=0)


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
            nn.Linear(edge_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.drop = nn.Dropout(dropout)

        self.use_global_attention = use_global_attention
        if use_global_attention:
            self.transformer = GraphTransformerLayer(hidden_dim, num_attention_heads, dropout)

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
    """Maps H → x_pred, predicting (Δva, Δvm) relative to flat start."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2)
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
        return torch.cat([va_init + delta[:, 0], vm_init + delta[:, 1]], dim=0)


class RegularizerHead(nn.Module):
    """Predicts per-graph Λ = diag(d) + U diag(s) Uᵀ."""

    def __init__(self, hidden_dim: int = 128, rank_k: int = 8):
        super().__init__()
        self.rank_k = rank_k
        self.d_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2)
        )
        self.u_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2 * rank_k)
        )
        self.s_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, rank_k)
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
    """Graph-level binary classifier: feasible (logit > 0) vs infeasible."""

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
# SOLVER HELPERS
# ============================================================================
def _solve_regularised_step(
    J: torch.Tensor,
    F_val: torch.Tensor,
    d: torch.Tensor,
    U: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    dim = J.shape[0]
    k = U.shape[1]
    D_sqrt = torch.diag(d.sqrt())
    V = torch.diag(s.sqrt()) @ U.T
    A = torch.cat([J, D_sqrt, V], dim=0)
    b = torch.cat([-F_val, torch.zeros(dim + k, device=J.device, dtype=J.dtype)])
    return torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)


def _solve_lm_step(
    J: torch.Tensor,
    F_val: torch.Tensor,
    mu_sqrt: float,
) -> torch.Tensor:
    dim = J.shape[0]
    device, dtype = J.device, J.dtype
    Lambda_sqrt = mu_sqrt * torch.eye(dim, device=device, dtype=dtype)
    A = torch.cat([J, Lambda_sqrt], dim=0)
    b = torch.cat([-F_val, torch.zeros(dim, device=device, dtype=dtype)])
    return torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)


def _project_voltage(x: torch.Tensor, vm_min: float, vm_max: float) -> torch.Tensor:
    """Clamp voltage magnitudes to [vm_min, vm_max]."""
    n = x.shape[0] // 2
    va = x[:n]
    vm = x[n:].clamp(min=vm_min, max=vm_max)
    return torch.cat([va, vm])


# ============================================================================
# UNROLLED SOLVER (BPTT only)
# ============================================================================
class UnrolledSolver(nn.Module):
    """Differentiable Newton-Raphson with back-propagation through time."""

    def __init__(self, T: int = 5, epsilon: float = 1e-7, vm_min: float = 0.5, vm_max: float = 1.5):
        super().__init__()
        self.T = T
        self.epsilon = epsilon
        self.vm_min = vm_min
        self.vm_max = vm_max

    # ------------------------------------------------------------------ #
    #  Learned regulariser — batched by same-size graph groups            #
    # ------------------------------------------------------------------ #
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

            for _t in range(self.T):
                # Force float32 for linear algebra regardless of AMP
                with torch.amp.autocast("cuda", enabled=False), torch.amp.autocast("cpu", enabled=False):
                    x_f32 = x.float()

                    F_list, J_list = [], []
                    for b in range(Bg):
                        gi = g_infos[b]
                        F_val = PowerFlowPhysics.compute_mismatch_from_x(
                            x_f32[b], gi["edge_index"], gi["edge_attr_raw"],
                            gi["p_spec"], gi["q_spec"],
                            gi["gs"], gi["bs"], gi["bus_type"], gi["vm_setpoint"],
                        )
                        J_val = PowerFlowPhysics.compute_jacobian(
                            x_f32[b], gi["edge_index"], gi["edge_attr_raw"],
                            gi["gs"], gi["bs"], gi["bus_type"], gi["vm_setpoint"],
                        )
                        F_list.append(F_val)
                        J_list.append(J_val)

                    F_batch = torch.stack(F_list)
                    max_res = F_batch.detach().abs().amax(dim=1).max().item()
                    if max_res < self.epsilon:
                        break

                    J_batch = torch.stack(J_list)
                    A = torch.cat([J_batch, D_sqrt_b.float(), V_b.float()], dim=1)
                    b_rhs = torch.cat([
                        -F_batch,
                        torch.zeros(Bg, dim + k, device=device, dtype=torch.float32),
                    ], dim=1)
                    dx = torch.linalg.lstsq(A, b_rhs.unsqueeze(-1)).solution.squeeze(-1)

                x = x + dx.to(x.dtype)
                x = torch.stack([_project_voltage(x[b], self.vm_min, self.vm_max) for b in range(Bg)])

            for b, idx in enumerate(indices):
                x_final_out[idx] = x[b]
                residuals_out[idx] = [F_list[b].detach().abs().max().item()] if F_list else []

        return x_final_out, residuals_out  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    #  Adaptive LM — batched, mu detached                                 #
    # ------------------------------------------------------------------ #
    def forward_batch_adaptive_lm(
        self,
        x_pred_list: List[torch.Tensor],
        per_graph: List[Dict[str, torch.Tensor]],
        cfg: Config,
    ) -> Tuple[List[torch.Tensor], List[List[float]], List[float]]:
        B = len(x_pred_list)
        device = x_pred_list[0].device

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
            mu = torch.full((Bg,), cfg.mu_init, device=device, dtype=torch.float32)
            g_infos = [per_graph[i] for i in indices]

            for _t in range(self.T):
                with torch.amp.autocast("cuda", enabled=False), torch.amp.autocast("cpu", enabled=False):
                    x_f32 = x.float()
                    F_list, J_list = [], []
                    for b in range(Bg):
                        gi = g_infos[b]
                        F_val = PowerFlowPhysics.compute_mismatch_from_x(
                            x_f32[b], gi["edge_index"], gi["edge_attr_raw"],
                            gi["p_spec"], gi["q_spec"],
                            gi["gs"], gi["bs"], gi["bus_type"], gi["vm_setpoint"],
                        )
                        J_val = PowerFlowPhysics.compute_jacobian(
                            x_f32[b], gi["edge_index"], gi["edge_attr_raw"],
                            gi["gs"], gi["bs"], gi["bus_type"], gi["vm_setpoint"],
                        )
                        F_list.append(F_val)
                        J_list.append(J_val)

                    F_batch = torch.stack(F_list)
                    max_res = F_batch.detach().abs().amax(dim=1).max().item()
                    if max_res < self.epsilon:
                        break

                    J_batch = torch.stack(J_list)

                    mu_sqrt = mu.detach().sqrt().view(Bg, 1, 1)
                    eye = torch.eye(dim, device=device, dtype=torch.float32).unsqueeze(0)
                    Lambda_sqrt = mu_sqrt * eye.expand(Bg, -1, -1)
                    A = torch.cat([J_batch, Lambda_sqrt], dim=1)
                    b_rhs = torch.cat([
                        -F_batch,
                        torch.zeros(Bg, dim, device=device, dtype=torch.float32),
                    ], dim=1)
                    dx = torch.linalg.lstsq(A, b_rhs.unsqueeze(-1)).solution.squeeze(-1)

                x_new = x + dx.to(x.dtype)
                x_new = torch.stack([
                    _project_voltage(x_new[b], self.vm_min, self.vm_max) for b in range(Bg)
                ])

                with torch.no_grad():
                    new_res_list = []
                    for b in range(Bg):
                        gi = g_infos[b]
                        Fn = PowerFlowPhysics.compute_mismatch_from_x(
                            x_new[b].float(), gi["edge_index"], gi["edge_attr_raw"],
                            gi["p_spec"], gi["q_spec"],
                            gi["gs"], gi["bs"], gi["bus_type"], gi["vm_setpoint"],
                        )
                        new_res_list.append(Fn.abs().max())
                    new_res = torch.stack(new_res_list)
                    old_res = F_batch.detach().abs().amax(dim=1)
                    improved = new_res < old_res
                    mu = torch.where(
                        improved,
                        (mu * cfg.mu_decrease).clamp(min=cfg.mu_min),
                        (mu * cfg.mu_increase).clamp(max=cfg.mu_max),
                    )

                x = torch.where(improved.unsqueeze(1), x_new, x)

            for b, idx in enumerate(indices):
                x_final_out[idx] = x[b]
                mu_out[idx] = mu[b].item()
                if F_list:
                    residuals_out[idx] = [F_list[b].detach().abs().max().item()]

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
            self.reg_head = RegularizerHead(hidden_dim=cfg.hidden_dim, rank_k=cfg.rank_k)
        else:
            self.reg_head = None

        self.solver = UnrolledSolver(
            T=cfg.T, epsilon=cfg.epsilon, vm_min=cfg.vm_min, vm_max=cfg.vm_max
        )

        self.register_buffer("x_mean", norm_stats["x_mean"])
        self.register_buffer("x_std", norm_stats["x_std"])
        self.register_buffer("edge_mean", norm_stats["edge_mean"])
        self.register_buffer("edge_std", norm_stats["edge_std"])

    # ------------------------------------------------------------------ #
    def _extract_per_graph(
        self, data: Batch, node_raw: Dict[str, torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
        edge_raw = denormalize_edge_features(data.edge_attr, self.edge_mean, self.edge_std)
        batch_idx = data.batch
        edge_src = data.edge_index[0]
        edge_dst = data.edge_index[1]

        _, counts = torch.unique_consecutive(batch_idx, return_counts=True)
        offsets = torch.zeros(counts.shape[0] + 1, dtype=torch.long, device=counts.device)
        offsets[1:] = counts.cumsum(0)

        per_graph: List[Dict[str, torch.Tensor]] = []
        for g_idx in range(counts.shape[0]):
            lo = offsets[g_idx].item()
            hi = offsets[g_idx + 1].item()
            n_g = hi - lo

            edge_mask = (
                (edge_src >= lo) & (edge_src < hi) & (edge_dst >= lo) & (edge_dst < hi)
            )
            ei_g = data.edge_index[:, edge_mask] - lo
            ea_g = {k: v[edge_mask] for k, v in edge_raw.items()}

            # Filter to forward edges for physics (avoid double-counting)
            if self.cfg.bidirectional_edges:
                src_g, dst_g = ei_g
                fwd = src_g < dst_g
                ei_phys = ei_g[:, fwd]
                ea_phys = {k: v[fwd] for k, v in ea_g.items()}
            else:
                ei_phys = ei_g
                ea_phys = ea_g

            sl = lo
            per_graph.append(
                {
                    "edge_index": ei_phys,
                    "edge_attr_raw": ea_phys,
                    "p_spec": node_raw["pg"][sl:hi] - node_raw["pd"][sl:hi],
                    "q_spec": -node_raw["qd"][sl:hi],
                    "gs": node_raw["gs"][sl:hi],
                    "bs": node_raw["bs"][sl:hi],
                    "bus_type": node_raw["bus_type"][sl:hi],
                    "vm_setpoint": node_raw["vm_setpoint"][sl:hi],
                    "n": n_g,
                }
            )
        return per_graph

    # ------------------------------------------------------------------ #
    def forward(
        self, data: Batch, run_solver: bool = True
    ) -> Dict[str, object]:
        H = self.encoder(data.x, data.edge_index, data.edge_attr, batch=data.batch)
        node_raw = denormalize_node_features(data.x, self.x_mean, self.x_std)

        x_pred = self.state_head(H, node_raw["bus_type"], node_raw["vm_setpoint"])
        infeas_logits = self.infeasibility_head(H, data.batch)

        per_graph = self._extract_per_graph(data, node_raw)
        total_nodes = data.x.shape[0]

        # split batch-level x_pred → per-graph list
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
                    self.solver.forward_batch_adaptive_lm(x_pred_list, per_graph, self.cfg)
                )
            else:
                x_final_list = x_pred_list
                all_residuals = [[] for _ in per_graph]
        else:
            reg_params = self.reg_head(H, data.batch)
            if run_solver:
                x_final_list, all_residuals = self.solver.forward_batch_regularised(
                    x_pred_list, reg_params, per_graph
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
        return torch.tensor(0.0, device=x_pred.device, requires_grad=True)
    return (
        (va_pred - va_true)[node_feasible].pow(2).sum()
        + (vm_pred - vm_true)[node_feasible].pow(2).sum()
    ) / (2.0 * n_feas)


def loss_physics(
    x_final_list: List[torch.Tensor],
    per_graph: List[Dict[str, torch.Tensor]],
    feasible_mask: torch.Tensor,
) -> torch.Tensor:
    total = torch.tensor(0.0, device=x_final_list[0].device)
    count = 0
    for i, (x_f, gi) in enumerate(zip(x_final_list, per_graph)):
        if not feasible_mask[i]:
            continue
        F_val = PowerFlowPhysics.compute_mismatch_from_x(
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
        total = total + F_val.pow(2).sum()
        count += F_val.shape[0]
    if count == 0:
        return torch.tensor(0.0, device=x_final_list[0].device, requires_grad=True)
    return total / count


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
    return F_func.binary_cross_entropy_with_logits(logits, feasible_mask.float())


def composite_loss(
    outputs: Dict,
    data: Batch,
    cfg: Config,
    stage: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    total_nodes = data.x.shape[0]
    feasible = data.feasible_mask

    l_s = loss_state(outputs["x_pred"], data.y_state, feasible, data.batch, total_nodes)
    l_i = loss_infeasibility(outputs["infeasibility_logits"], feasible)

    metrics: Dict[str, float] = {"L_state": l_s.item(), "L_infeas": l_i.item()}

    if stage == 1:
        total = cfg.lambda_1 * l_s + cfg.lambda_infeasibility * l_i
        metrics["L_total"] = total.item()
        return total, metrics

    l_p = loss_physics(outputs["x_final_list"], outputs["per_graph"], feasible)
    metrics["L_phys"] = l_p.item()

    total = cfg.lambda_1 * l_s + cfg.lambda_2 * l_p + cfg.lambda_infeasibility * l_i

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
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

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
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        setup_file_logging(cfg.log_dir)

        if _WANDB_AVAILABLE:
            wandb.init(project=cfg.wandb_project, config=asdict(cfg), reinit=True)
            wandb.watch(model, log="gradients", log_freq=100)

        self.cumulative_train_time = 0.0

        # per-stage early stopping
        self.best_val_loss = float("inf")
        self.best_model_state: Optional[Dict] = None
        self.es_counter = 0

    def _save_checkpoint(self, epoch: int, stage: int, tag: str = ""):
        name = f"epoch_{epoch}_stage{stage}{tag}.pt"
        path = os.path.join(self.cfg.checkpoint_dir, name)
        torch.save(
            {
                "epoch": epoch,
                "stage": stage,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            path,
        )
        log.info("Checkpoint → %s", path)

    @torch.no_grad()
    def _validate(self, stage: int) -> Dict[str, float]:
        self.model.eval()
        running: Dict[str, float] = {}
        count = 0
        for data in self.val_loader:
            data = data.to(self.device)
            outputs = self.model(data, run_solver=(stage == 2))
            _, metrics = composite_loss(outputs, data, self.cfg, stage)
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v
            count += 1
        return {k: v / max(count, 1) for k, v in running.items()}

    def _run_stage(self, stage: int, num_epochs: int, epoch_offset: int):
        log.info("=" * 60)
        log.info("STAGE %d — %d epochs  (solver %s)", stage, num_epochs,
                 "ON" if stage == 2 else "OFF")
        log.info("=" * 60)

        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.es_counter = 0

        for epoch_local in range(num_epochs):
            epoch_global = epoch_offset + epoch_local
            self.model.train()
            epoch_loss = 0.0
            epoch_metrics: Dict[str, float] = {}
            n_batches = 0
            t0 = time.time()

            for data in self.train_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                    outputs = self.model(data, run_solver=(stage == 2))
                    loss, metrics = composite_loss(outputs, data, self.cfg, stage)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()
                for k, v in metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
                n_batches += 1

            t_epoch = time.time() - t0
            self.cumulative_train_time += t_epoch
            avg_loss = epoch_loss / max(n_batches, 1)
            avg_metrics = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}

            self.scheduler.step(avg_loss)

            val_metrics = self._validate(stage)
            val_total = val_metrics.get("L_total", val_metrics.get("L_state", float("inf")))

            # early stopping
            if val_total < self.best_val_loss:
                self.best_val_loss = val_total
                self.best_model_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                self.es_counter = 0
            else:
                self.es_counter += 1

            log_dict = {
                "stage": stage,
                "epoch": epoch_global,
                "train/loss": avg_loss,
                "train/time_s": t_epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            for k, v in avg_metrics.items():
                log_dict[f"train/{k}"] = v
            for k, v in val_metrics.items():
                log_dict[f"val/{k}"] = v
            if _WANDB_AVAILABLE and wandb.run is not None:
                wandb.log(log_dict, step=epoch_global)

            summary = (
                f"loss={avg_loss:.5f}  val={val_total:.5f}  "
                f"t={t_epoch:.1f}s  lr={self.optimizer.param_groups[0]['lr']:.2e}  "
                f"es={self.es_counter}/{self.cfg.early_stop_patience}"
            )
            log.info("Ep %d (stage %d)  %s", epoch_global + 1, stage, summary)

            if (epoch_global + 1) % 10 == 0:
                self._save_checkpoint(epoch_global + 1, stage)

            if self.es_counter >= self.cfg.early_stop_patience:
                log.info("Early stopping triggered at epoch %d", epoch_global + 1)
                break

        # restore best
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            log.info("Restored best model (val_loss=%.6f)", self.best_val_loss)

    def train(self, stages: str = "both"):
        t_total = time.time()
        run_s1 = stages in ("1", "both")
        run_s2 = stages in ("2", "both")

        if run_s1:
            self._run_stage(1, self.cfg.epochs_stage1, 0)
            self._save_checkpoint(self.cfg.epochs_stage1, 1, tag="_final")

        if run_s2:
            offset = self.cfg.epochs_stage1 if run_s1 else 0
            self._run_stage(2, self.cfg.epochs_stage2, offset)

        total_time = time.time() - t_total
        log.info("Training complete in %.1fs", total_time)

        final_path = os.path.join(self.cfg.checkpoint_dir, "final_model.pt")
        torch.save(self.model.state_dict(), final_path)
        log.info("Final model → %s", final_path)


# ============================================================================
# INFERENCE
# ============================================================================
class InferenceEngine:
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device

    @torch.no_grad()
    def run(
        self, model: BifurcationAwarePFSolver, dataloader: DataLoader
    ) -> Dict[str, float]:
        model.eval()
        orig_T = model.solver.T
        model.solver.T = self.cfg.max_iter_inference

        all_converged: List[bool] = []
        all_pred_infeasible: List[bool] = []
        all_true_feasible: List[bool] = []
        total_samples = 0
        total_time = 0.0

        for data in tqdm(dataloader, desc="Inference", unit="batch"):
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
                    stagnated = abs(residuals[-1] - residuals[-2]) < self.cfg.stagnation_tol

                # learned classifier signal
                pred_infeas_learned = infeas_logits[i].item() < 0.0

                # heuristic signal
                if self.cfg.use_adaptive_lm:
                    mu_f = (
                        outputs["final_mu_list"][i]
                        if i < len(outputs.get("final_mu_list", []))
                        else self.cfg.mu_init
                    )
                    heuristic_flag = (not converged) and stagnated and (mu_f > self.cfg.tau)
                elif outputs.get("reg_params"):
                    d_g, _, s_g = outputs["reg_params"][i]
                    tr_lam = d_g.sum().item() + s_g.sum().item()
                    heuristic_flag = (not converged) and stagnated and (tr_lam > self.cfg.tau)
                else:
                    heuristic_flag = (not converged) and stagnated

                pred_infeasible = pred_infeas_learned or heuristic_flag

                all_converged.append(converged)
                all_pred_infeasible.append(pred_infeasible)
                all_true_feasible.append(bool(data.feasible_mask[i].item()))

        conv_rate = sum(all_converged) / max(total_samples, 1)
        true_infeasible = [not f for f in all_true_feasible]
        tp = sum(p and t for p, t in zip(all_pred_infeasible, true_infeasible))
        fp = sum(p and (not t) for p, t in zip(all_pred_infeasible, true_infeasible))
        fn = sum((not p) and t for p, t in zip(all_pred_infeasible, true_infeasible))
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
            total_samples, total_time, per_sample, conv_rate, prec, rec, f1,
        )

        model.solver.T = orig_T
        return metrics


# ============================================================================
# MAIN
# ============================================================================
def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bifurcation-Aware PF Solver")
    p.add_argument("--data-dir", default="data/processed/task4_solvability")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--wandb-project", default="pfdelta-bifurcation")

    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-mp-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-global-attention", action="store_true")
    p.add_argument("--num-attention-heads", type=int, default=4)
    p.add_argument("--rank-k", type=int, default=8)

    p.add_argument("--T", type=int, default=5)
    p.add_argument("--epsilon", type=float, default=1e-7)
    p.add_argument("--max-iter-inference", type=int, default=20)

    p.add_argument("--adaptive-lm", action="store_true")
    p.add_argument("--mu-init", type=float, default=1e-3)
    p.add_argument("--mu-min", type=float, default=1e-8)
    p.add_argument("--mu-max", type=float, default=1e6)
    p.add_argument("--mu-decrease", type=float, default=0.5)
    p.add_argument("--mu-increase", type=float, default=2.0)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs-stage1", type=int, default=50)
    p.add_argument("--epochs-stage2", type=int, default=100)
    p.add_argument("--lambda-1", type=float, default=1.0)
    p.add_argument("--lambda-2", type=float, default=1.0)
    p.add_argument("--lambda-3", type=float, default=1e-3)
    p.add_argument("--lambda-infeasibility", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=25)

    p.add_argument("--tau", type=float, default=50.0)
    p.add_argument("--stagnation-tol", type=float, default=1e-6)
    p.add_argument("--vm-min", type=float, default=0.5)
    p.add_argument("--vm-max", type=float, default=1.5)

    p.add_argument("--unidirectional-edges", action="store_true",
                   help="Set if each branch is stored as ONE directed edge.")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--stage", choices=["1", "2", "both"], default="both")
    p.add_argument("--resume-from", default=None)
    p.add_argument("--eval-only", default=None)
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
        tau=args.tau,
        stagnation_tol=args.stagnation_tol,
        vm_min=args.vm_min,
        vm_max=args.vm_max,
        bidirectional_edges=not args.unidirectional_edges,
        use_amp=args.amp,
        seed=args.seed,
    )

    seed_everything(cfg.seed)
    device = torch.device("cpu") if (args.cpu or not torch.cuda.is_available()) else torch.device("cuda")
    log.info("Device: %s", device)

    train_loader, val_loader, test_loader, norm_stats = load_datasets(cfg)
    log.info(
        "Data — train: %d  val: %d  test: %d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    model = BifurcationAwarePFSolver(cfg, norm_stats)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Parameters: %d  |  adaptive_lm=%s  global_attn=%s  bidir_edges=%s",
             n_params, cfg.use_adaptive_lm, cfg.use_global_attention, cfg.bidirectional_edges)

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
        setup_file_logging(cfg.log_dir)
        _load_ckpt(args.eval_only)
        model = model.to(device)
        if _WANDB_AVAILABLE:
            wandb.init(project=cfg.wandb_project, config=asdict(cfg), reinit=True)
        engine = InferenceEngine(cfg, device)
        engine.run(model, test_loader)
        if _WANDB_AVAILABLE:
            wandb.finish()
        return

    if args.resume_from:
        _load_ckpt(args.resume_from)
    elif args.stage == "2":
        log.warning("Stage 2 without --resume-from: starting from random init.")

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