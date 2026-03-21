from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from ..base import build_mlp, ensure_2d_float_tensor


class MLPEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_sizes: Sequence[int],
        activation_cls: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_sizes = tuple(hidden_sizes)
        self.output_dim = self.hidden_sizes[-1] if self.hidden_sizes else input_dim
        self.network = build_mlp(input_dim, self.hidden_sizes, activation_cls=activation_cls)

    def forward(self, observations: Tensor) -> Tensor:
        return self.network(ensure_2d_float_tensor(observations))
