import torch
import torch.nn as nn
import torch.nn.functional as F

from data_norm import RMSNorm

class MLPLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        base_activation="silu",
        norm_type="layer",
        use_attn=False,
    ):
        super().__init__()

        self.base_activation = base_activation
        self.use_attn = use_attn

        self.linear = nn.Linear(input_dim, output_dim, bias=True)

        if norm_type == "layer":
            self.norm = nn.LayerNorm(input_dim)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm1d(input_dim)
        elif norm_type == "rms":
            self.norm = RMSNorm(input_dim)
        elif norm_type == "mm":    
            self.norm = MMNorm(input_dim)
        else:
            self.norm = nn.Identity()

        if use_attn:
            self.attn_proj = nn.Linear(input_dim, input_dim)
            self.temperature = nn.Parameter(
                torch.tensor(input_dim**0.5)
            )

    def activation(self, x):
        activation_funcs = {
            "softplus": F.softplus,
            "sigmoid": torch.sigmoid,
            "silu": F.silu,
            "relu": F.relu,
            "leaky_relu": F.leaky_relu,
            "elu": F.elu,
            "gelu": F.gelu,
            "selu": F.selu,
            "tanh": torch.tanh,
            "sine": torch.sin,
        }
        return activation_funcs.get(self.base_activation, lambda x: x)(x)

    def global_attn(self, x):
        scores = self.attn_proj(x)
        weights = F.softmax(
            scores / self.temperature.clamp(min=1.0),
            dim=-1,
        )
        return x * weights

    def forward(self, x):
        if self.use_attn:
            x = self.global_attn(x)

        x = self.norm(x)
        x = self.linear(x)
        x = self.activation(x)

        return x


class MLP(nn.Module):
    def __init__(
        self,
        net_layers,
        base_activation="silu",
        norm_type="layer",
        use_attn=False,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            MLPLayer(
                in_dim,
                out_dim,
                base_activation,
                norm_type,
                use_attn,
            )
            for in_dim, out_dim in zip(
                net_layers[:-1],
                net_layers[1:],
            )
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x