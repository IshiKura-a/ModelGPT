from torch import nn, Tensor


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            *[
                nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
            ],
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)
