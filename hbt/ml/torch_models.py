from __future__ import annotations

__all__ = [
]

from collections import Iterable
from columnflow.util import MockModule, maybe_import, DotDict
from columnflow.types import T, Any, Callable, Sequence
from columnflow.columnar_util import Route, EMPTY_FLOAT, EMPTY_INT
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
            columns = [
                "leptons.{px,py,pz,energy,mass}[:, 0]",
                "leptons.{px,py,pz,energy,mass}[:, 1]",
                "bjets.{px,py,pz,energy,mass,btagDeepFlavB,btagDeepFlavCvB,btagDeepFlavCvL,hhbtag}[:, 0]",
                "bjets.{px,py,pz,energy,mass,btagDeepFlavB,btagDeepFlavCvB,btagDeepFlavCvL,hhbtag}[:, 1]",
                "fatjets.{px,py,pz,energy,mass}[:, 0]",
            ]
            self.inputs = set()
            self.inputs.update(*list(map(Route, law.util.brace_expand(obj)) for obj in columns))

            self.linear_relu_stack = nn.Sequential(
                nn.BatchNorm1d(len(self.inputs),),
                nn.Linear(len(self.inputs), 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
            )

        def forward(self, x):
            input_data = x
            if isinstance(x, dict):
                input_data: torch.Tensor = torch.cat([
                    val.reshape(-1, 1) for key, val in x.items()
                    if key in [str(r) for r in self.inputs]],
                    axis=-1
                )
            # check for dummy values
            empty_float = input_data == EMPTY_FLOAT
            empty_int = input_data == EMPTY_INT
            input_data[empty_float | empty_int] = 0
            logits = self.linear_relu_stack(input_data.to(torch.float32))
            return logits
        
