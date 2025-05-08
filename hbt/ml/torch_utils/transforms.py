from __future__ import annotations

from collections.abc import Mapping
from columnflow.util import MockModule, maybe_import
from columnflow.columnar_util import flat_np_view
from hbt.production.res_networks import res_net_preprocessing

torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
np = maybe_import("numpy")
ak = maybe_import("awkward")


if not isinstance(torch, MockModule):
    from torch.nested._internal.nested_tensor import NestedTensor

    class AkToNestedTensor(torch.nn.Module):

        def __init__(self, requires_grad=False, device=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.requires_grad = requires_grad
            self.device = device

            # Dict of NumPy dtype -> torch dtype (when the correspondence exists)
            # from https://github.com/pytorch/pytorch/blob/e180ca652f8a38c479a3eff1080efe69cbc11621/torch/testing/_internal/common_utils.py#L349  # noqa
            self.numpy_to_torch_dtype_dict = {
                np.bool: torch.bool,
                np.uint8: torch.uint8,
                np.int8: torch.int8,
                np.int16: torch.int16,
                np.int32: torch.int32,
                np.int64: torch.int64,
                np.float16: torch.float16,
                np.float32: torch.float32,
                np.float64: torch.float64,
                np.complex64: torch.complex64,
                np.complex128: torch.complex128,
            }

        def _transform_input(self, X):
            return_tensor = None
            if isinstance(X, ak.Array):
                # first, get flat numpy view to avoid copy operations
                view = flat_np_view(X)

                # transform into torch Tensor with same type
                values = torch.Tensor(view).to(
                    self.numpy_to_torch_dtype_dict.get(view.dtype.type, view.dtype.type),
                )

                # to calculate the offsets, count the elements
                n_elements = ak.num(X, axis=1)

                # the offsets are the cumulated number of elements
                # prepend 0 to also account for first element in array
                try:
                    cumsum = np.cumsum(ak.concatenate([0, n_elements], axis=0), axis=0)
                except Exception as e:
                    from IPython import embed
                    embed(header=f"raised Exception '{e}', debugging")
                    raise e

                # alternative way to get underlying offsets and data structure
                # array = ak.to_layout(X)
                # offsets = array.offsets.data
                # view = array.content.content.data
                # this should always yield the correct offsets, regardless of slicing

                # now directly construct NestedTensor
                # DANGERZONE: after comparing hex(id(values)) vs hex(id(return_tensor.values()))
                # realized that there must be a copy going on somewhere...
                return_tensor = torch.nested._internal.nested_tensor.NestedTensor(
                    values=values, offsets=torch.Tensor(cumsum),
                    requires_grad=self.requires_grad, device=self.device,
                )
            elif isinstance(X, torch.Tensor):
                # if the input is already a Tensor, cast it into a nested tensor
                return_tensor = torch.nested.as_nested_tensor(
                    X, layout=torch.jagged, device=self.device,
                )
            elif isinstance(X, NestedTensor):
                return_tensor = X

            if return_tensor is None:
                raise ValueError(f"Could not convert input {X=}")

            return return_tensor

        def forward(
            self,
            X: ak.Array | torch.Tensor | NestedTensor | Mapping[str, ak.Array | torch.Tensor | NestedTensor],
        ) -> (NestedTensor | Mapping[str, NestedTensor]):
            return_tensor: NestedTensor | Mapping[str, NestedTensor] | None = None

            if isinstance(X, Mapping):
                return_tensor = {
                    key: self._transform_input(data) for key, data in X.items()
                }
            else:
                return_tensor = self._transform_input(X)

            return return_tensor

    class MoveToDevice(torch.nn.Module):
        def __init__(self, device=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.device = device

        def forward(self, X: ak.Array | torch.Tensor | NestedTensor) -> (ak.Array | torch.Tensor | NestedTensor):
            if isinstance(X, (ak.Array, np.ndarray)):
                return ak.to_torch(X).to(device=self.device)
            elif isinstance(X, (torch.Tensor, NestedTensor)):
                return X.to(device=self.device)
            elif isinstance(X, (list, tuple)):
                return [self.forward(entry) for entry in X]
            else:
                raise ValueError(f"Could not convert input {X=}")

    class AkToTensor(AkToNestedTensor):
        def _transform_input(self, X):
            return_tensor = None
            if isinstance(X, (ak.Array, np.ndarray)):
                # first, get flat numpy view to avoid copy operations
                return_tensor = ak.to_torch(X).to(device=self.device)

            elif isinstance(X, (torch.Tensor, NestedTensor)):
                return_tensor = X
            elif isinstance(X, (list, tuple)):
                return_tensor = [self._transform_input(entry) for entry in X]
            elif isinstance(X, dict):
                return_tensor = {key: self._transform_input(val) for key, val in X.items()}

            if return_tensor is None:
                raise ValueError(f"Could not convert input {X=}")

            return return_tensor

    class EventFilter(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            self.uses = {"category_ids"}

    class PreProcessFloatValues(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.preprocessing = res_net_preprocessing()
            self.uses = self.preprocessing.used_columns
            self.produces = self.preprocessing.produced_columns

        def forward(self, events: ak.Array) -> ak.Array:
            events = self.preprocessing(events)
            return events

    class PreProssesAndCast(torch.nn.Module):
        def __init__(self, *args, device=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.device = device
            self.preprocess = PreProcessFloatValues()
            self.cast = AkToTensor(device=self.device)

        def forward(self, events: ak.Array) -> torch.Tensor:
            x = self.preprocess(events)
            return self.cast(x)

        def _get_property(self, attr="uses"):
            attr_set = set()
            for t in [self.preprocess, self.cast]:
                if hasattr(t, attr):
                    attr_set.update(getattr(t, attr))
            return attr_set

        @property
        def uses(self):
            return self._get_property("uses")

        @property
        def produces(self):
            return self._get_property("produces")
