from __future__ import annotations

import warnings
from collections import defaultdict

from columnflow.util import MockModule, maybe_import

torch = maybe_import("torch")
ignite = maybe_import("ignite")
np = maybe_import("numpy")

from columnflow.types import Callable, Any, Sequence
from typing import cast

if not isinstance(torch, MockModule):
    from ignite.metrics.epoch_metric import EpochMetric, EpochMetricWarning
    from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce
    from ignite.metrics.loss import Loss
    from ignite.exceptions import NotComputableError
    import ignite.distributed as idist

    class WeightedEpochMetric(EpochMetric):
        _state_dict_all_req_keys = ("_predictions", "_targets")

        @reinit__is_reduced
        def reset(self) -> None:
            self._predictions: list[torch.Tensor] = []
            self._targets: list[torch.Tensor] = []
            self._result: float | None = None
            self._kwargs: defaultdict[str, list[Any]] = defaultdict(list)

        def _check_shape(
            self,
            output: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]],
        ) -> None:
            y_pred, y = output[:2]
            return super()._check_shape((y_pred, y))

        def _check_type(
            self,
            output: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]],
        ) -> None:
            y_pred, y = output[:2]
            return super()._check_type((y_pred, y))

        @reinit__is_reduced
        def update(
            self,
            output: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]],
        ) -> None:
            self._check_shape(output)
            y_pred, y = output[0].detach(), output[1].detach()
            kwargs = dict()
            if len(output) == 3:
                kwargs = output[2]

            def fix_dimensions(tensor: torch.Tensor) -> torch.Tensor:
                if tensor.ndimension() == 2 and tensor.shape[1] == 1:
                    return tensor.squeeze(dim=-1)
                return tensor.clone().to(self._device)

            y_pred = fix_dimensions(y_pred)
            y = fix_dimensions(y)
            kwargs = {k: fix_dimensions(v) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

            self._check_type((y_pred, y))
            self._predictions.append(y_pred)
            self._targets.append(y)
            for k, v in kwargs.items():
                self._kwargs[k].append(v)

            # Check once the signature and execution of compute_fn
            if len(self._predictions) == 1 and self._check_compute_fn:
                try:
                    self.compute_fn(self._predictions[0], self._targets[0])
                except Exception as e:
                    warnings.warn(f"Probably, there can be a problem with `compute_fn`:\n {e}.", EpochMetricWarning)

        def compute(self) -> float:
            if len(self._predictions) < 1 or len(self._targets) < 1:
                raise NotComputableError("EpochMetric must have at least one example before it can be computed.")

            if self._result is None:
                _prediction_tensor = torch.cat(self._predictions, dim=0)
                _target_tensor = torch.cat(self._targets, dim=0)
                _kwargs = dict()
                for k, v in self._kwargs.items():
                    example = v[0]
                    if isinstance(example, torch.Tensor):
                        _kwargs[k] = torch.cat(v, dim=0)
                    else:
                        if not all(x == example for x in v):
                            raise ValueError(f"All elements of {k} must be the same.")
                        _kwargs[k] = example

                ws = idist.get_world_size()
                if ws > 1:
                    # All gather across all processes
                    _prediction_tensor = cast(torch.Tensor, idist.all_gather(_prediction_tensor))
                    _target_tensor = cast(torch.Tensor, idist.all_gather(_target_tensor))
                    for k, v in _kwargs.items():
                        if isinstance(v, torch.Tensor):
                            _kwargs[k] = cast(torch.Tensor, idist.all_gather(v))

                self._result = 0.0
                if idist.get_rank() == 0:
                    # Run compute_fn on zero rank only
                    self._result = self.compute_fn(_prediction_tensor, _target_tensor, **_kwargs)

                if ws > 1:
                    # broadcast result to all processes
                    self._result = cast(float, idist.broadcast(self._result, src=0))

            return self._result

    def roc_auc_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor, **kwargs) -> float:
        from sklearn.metrics import roc_auc_score

        y_true = y_targets.cpu().numpy()
        y_pred = y_preds.cpu().numpy()
        numpy_kwargs = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        if "weight" in numpy_kwargs:
            # If weights are provided, we need to convert them to a 1D array
            numpy_kwargs["sample_weight"] = numpy_kwargs.pop("weight")
        result = roc_auc_score(y_true, y_pred, **numpy_kwargs)

        return result

    class WeightedROC_AUC(WeightedEpochMetric):
        def __init__(
            self,
            output_transform: Callable = lambda x: x,
            check_compute_fn: bool = False,
            device: str | torch.device = torch.device("cpu"),
            skip_unrolling: bool = False,
            target_class_idx: int | None = None,
        ) -> None:
            try:
                from sklearn.metrics import roc_auc_score  # noqa: F401
            except ImportError:
                raise ModuleNotFoundError("This contrib module requires scikit-learn to be installed.")

            super().__init__(
                roc_auc_compute_fn,
                output_transform=output_transform,
                check_compute_fn=check_compute_fn,
                device=device,
                skip_unrolling=skip_unrolling,
            )
            self.target_class_idx = target_class_idx

        def compute(self):
            result = super().compute()
            if self.target_class_idx is not None:
                # If target_idx is provided, we need to filter the result
                # to only include the specified target index
                self._result = result[self.target_class_idx]
            return self._result

    class WeightedLoss(Loss):
        @reinit__is_reduced
        def reset(self) -> None:
            self._internal_sum = list()
            self._internal_num_examples = list()

        @reinit__is_reduced
        def update(self, output: Sequence[torch.Tensor | dict[str, Any]]) -> None:
            if len(output) == 2:
                y_pred, y = cast(tuple[torch.Tensor, torch.Tensor], output)
                kwargs: dict[str, Any] = {}
            else:
                y_pred, y, kwargs = cast(tuple[torch.Tensor, torch.Tensor, dict], output)
            average_loss = self._loss_fn(y_pred, y, **kwargs).detach()

            if len(average_loss.shape) != 0:
                raise ValueError("loss_fn did not return the average loss.")

            n = 1
            if "weight" in kwargs.keys():
                n = torch.sum(kwargs["weight"]).item()
            else:
                n = self._batch_size(y)
            self._internal_sum.append(average_loss.to(self._device) * n)
            self._internal_num_examples.append(n)

        @property
        def _sum(self):
            return torch.sum(torch.tensor(self._internal_sum))

        @property
        def _num_examples(self):
            return torch.sum(torch.tensor(self._internal_num_examples))

        @sync_all_reduce("_sum", "_num_examples")
        def compute(self) -> float:
            if self._num_examples == 0:
                raise NotComputableError("Loss must have at least one example before it can be computed.")
            return self._sum.item() / self._num_examples
