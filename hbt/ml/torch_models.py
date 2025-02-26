from __future__ import annotations

__all__ = [
]

from collections import Iterable
from columnflow.util import MockModule, maybe_import, DotDict
from columnflow.types import T, Any, Callable, Sequence
from columnflow.columnar_util import Route
torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
np = maybe_import("numpy")
ak = maybe_import("awkward")

if not isinstance(torch, MockModule):
    from torch import nn

    class FeedForwardNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )
            inputs = ["Electron.pt", "Muon.pt", "Jet.pt"]

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits
        
