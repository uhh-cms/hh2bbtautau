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
import law

if not isinstance(torch, MockModule):
    from torch import nn

    class FeedForwardNet(nn.Module):
        def __init__(self):
            super().__init__()
            columns = ["Jet.{pt,eta,phi}[:, 0]", "Muon.pt[:, 0]", "Electron.pt[:, 0]"]
            self.inputs = set()
            self.inputs.update(*list(map(Route, law.util.brace_expand(obj)) for obj in columns))

            self.linear_relu_stack = nn.Sequential(
                nn.Linear(len(self.inputs), 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
            )
            

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits
        
