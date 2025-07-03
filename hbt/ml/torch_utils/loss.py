from __future__ import annotations

__all__ = [
    "WeightedCrossEntropy",

]


from columnflow.columnar_util import EMPTY_INT, EMPTY_FLOAT, Route
from columnflow.util import maybe_import, MockModule

torch = maybe_import("torch")
WeightedCrossEntropy = MockModule("WeightedCrossEntropy")  # type: ignore[assignment]

if not isinstance(torch, MockModule):
    class WeightedCrossEntropy(torch.nn.CrossEntropyLoss):
        def forward(self, input, target, weight: torch.Tensor | None = None):
            # save original reduction mode
            reduction = self.reduction
            if weight is not None:
                self.reduction = "none"
                loss = super().forward(input, target)
                self.reduction = reduction

                # dot product is only defined for flat tensors, so flatten
                loss = torch.flatten(loss)
                weight = torch.flatten(weight)
                loss = torch.dot(loss, weight)
                if self.reduction == "mean":
                    loss = loss / torch.sum(weight)
            else:
                loss = super().forward(input, target)
            return loss
