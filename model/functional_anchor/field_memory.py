from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ellipse_sdf(size: int, rx: float, ry: float, cx: float = 0.0, cy: float = 0.0) -> torch.Tensor:
    coords = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    x = (xx - float(cx)) / max(float(rx), 1.0e-4)
    y = (yy - float(cy)) / max(float(ry), 1.0e-4)
    sdf = (1.0 - torch.sqrt(x.square() + y.square()).clamp_min(1.0e-6)) * 2.0
    return sdf.clamp(-2.0, 2.0)


class FieldMemory(nn.Module):
    """Function-code anchor field: code → basis decoder → canonical SDF.

    Maintains:
        anchor_keys           [A, query_dim]
        anchor_function_codes [A, code_dim]
        sdf_basis             [B_basis, 1, S, S]
        code_to_basis         code_dim → basis_dim
    """

    def __init__(
        self,
        num_anchors: int,
        query_dim: int,
        code_dim: int,
        basis_dim: int,
        anchor_size: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.num_anchors = int(num_anchors)
        self.code_dim = int(code_dim)
        self.basis_dim = int(basis_dim)
        self.anchor_size = int(anchor_size)

        self.anchor_keys = nn.Parameter(torch.randn(self.num_anchors, query_dim) * 0.02)
        nn.init.orthogonal_(self.anchor_keys[: min(num_anchors, query_dim)])

        self.anchor_function_codes = nn.Parameter(
            torch.randn(self.num_anchors, self.code_dim) * 0.02
        )
        self.anchor_basis_logits = nn.Parameter(
            torch.empty(self.num_anchors, self.basis_dim)
        )
        self.sdf_basis = nn.Parameter(
            torch.randn(self.basis_dim, 1, self.anchor_size, self.anchor_size) * 0.02
        )
        self.code_to_basis = nn.Sequential(
            nn.Linear(self.code_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.basis_dim),
        )

        with torch.no_grad():
            init_anchors = [
                _ellipse_sdf(anchor_size, 0.66, 0.82, 0.00, 0.00),
                _ellipse_sdf(anchor_size, 0.50, 0.66, 0.03, 0.00),
                _ellipse_sdf(anchor_size, 0.34, 0.46, 0.00, 0.02),
                _ellipse_sdf(anchor_size, 0.52, 0.58, -0.14, 0.07),
                _ellipse_sdf(anchor_size, 0.44, 0.72, 0.12, -0.05),
            ]
            while len(init_anchors) < self.num_anchors:
                idx = len(init_anchors)
                radius = max(0.30, 0.60 - 0.04 * idx)
                init_anchors.append(_ellipse_sdf(anchor_size, radius, radius * 1.2, 0.05 * ((idx % 3) - 1), 0.0))
            init_stack = torch.stack(init_anchors[: self.num_anchors]).unsqueeze(1)
            for b in range(self.basis_dim):
                self.sdf_basis[b] = init_stack[b % self.num_anchors]
            nn.init.zeros_(self.code_to_basis[-1].weight)
            nn.init.zeros_(self.code_to_basis[-1].bias)
            self.anchor_basis_logits.fill_(-2.0)
            basis_ramp = torch.linspace(-1.0, 1.0, self.basis_dim, device=self.anchor_basis_logits.device)
            for anchor_idx in range(self.num_anchors):
                self.anchor_basis_logits[anchor_idx, anchor_idx % self.basis_dim] = 2.0
                self.anchor_basis_logits[anchor_idx, (anchor_idx + 1) % self.basis_dim] = 0.5
                self.anchor_basis_logits[anchor_idx] += 0.05 * float(anchor_idx) * basis_ramp

    def decode_static_field(self) -> torch.Tensor:
        """Decode function codes → canonical SDF.  Call once per video forward.

        Returns:
            canonical_sdf: [A, 1, S, S]
        """
        basis_weights = self.get_basis_weights()
        return torch.einsum("nb,bchw->nchw", basis_weights, self.sdf_basis)

    def get_basis_weights(self) -> torch.Tensor:
        logits = self.anchor_basis_logits + self.code_to_basis(self.anchor_function_codes)
        return torch.softmax(logits, dim=-1)

    def code_pairwise_similarity(self) -> torch.Tensor:
        code_vectors = F.normalize(self.anchor_function_codes.detach(), dim=-1)
        pairwise = torch.matmul(code_vectors, code_vectors.transpose(0, 1))
        eye = torch.eye(self.num_anchors, device=code_vectors.device, dtype=torch.bool)
        return pairwise.masked_select(~eye).mean() if self.num_anchors > 1 else torch.zeros((), device=code_vectors.device)
