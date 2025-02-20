import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """This class implements the feed-forward network derived from Llama2.

    Args:
        gate_proj (nn.Module): Projection from input dim to hidden dim, fed through activation
            and multiplied by up_proj.
        down_proj (nn.Module): Final projection to output dim.
        up_proj (Optional[nn.Module]): Projection from input dim to hidden dim, multiplied by
            activation(gate_proj).
        activation (nn.Module): Activation function to use. Default is nn.SiLU().
    """

    def __init__(
        self,
        *,
        dim,            
        bias: bool = False,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, dim, bias=bias)
        self.w2 = nn.Linear(dim, dim, bias=bias)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape ``(..., in_dim)``, where ``in_dim`` is the
            input dimension of both ``gate_proj`` and ``up_proj``.

        Returns:
            torch.Tensor: output tensor with shape ``(..., out_dim)``, where ``out_dim`` is the \
            output dimension of ``down_proj``.
        """
        w1_out = self.w1(x)
        h = self.activation(w1_out)
        h = h * self.w3(x)
        h = self.w2(h)
        return h

class DeepFeedForward(nn.Module):
    def __init__(
        self,
        dim,
        depth: int,
        bias: bool = False,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FeedForward(
                    dim=dim, bias=bias, activation=activation
                )
                for _ in range(depth)
            ]
        )
        self.w4 = nn.Linear(dim, dim, bias=bias)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.w4(x)
        return x

class DummyModel(nn.Module):
    def __init__(self, dim: int,  depth: int):
        super().__init__()
        self.ff_1 = DeepFeedForward(dim=dim, depth=depth)
        self.ff_2 = DeepFeedForward(dim=dim, depth=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff_1(x)
        x = self.ff_2(x)
        return x