from __future__ import annotations

from collections.abc import Mapping, Collection
from columnflow.util import MockModule, maybe_import, DotDict
from columnflow.types import T, Any, Callable, Sequence
from columnflow.columnar_util import (
    flat_np_view, attach_behavior, default_coffea_collections, set_ak_column,
    Route, attach_coffea_behavior,
)
import copy
import traceback

torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
np = maybe_import("numpy")
ak = maybe_import("awkward")

if not isinstance(torch, MockModule):
    from torch.nested._internal.nested_tensor import NestedTensor
    from IPython import embed
    class AkToNestedTensor(torch.nn.Module):
        
        def __init__(self, requires_grad=False, device=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.requires_grad = requires_grad
            self.device = device

            # Dict of NumPy dtype -> torch dtype (when the correspondence exists)
            # from https://github.com/pytorch/pytorch/blob/e180ca652f8a38c479a3eff1080efe69cbc11621/torch/testing/_internal/common_utils.py#L349  # noqa
            self.numpy_to_torch_dtype_dict = {
                np.bool       : torch.bool,
                np.uint8      : torch.uint8,
                np.int8       : torch.int8,
                np.int16      : torch.int16,
                np.int32      : torch.int32,
                np.int64      : torch.int64,
                np.float16    : torch.float16,
                np.float32    : torch.float32,
                np.float64    : torch.float64,
                np.complex64  : torch.complex64,
                np.complex128 : torch.complex128
            }
        def _transform_input(self, X):
            return_tensor = None
            if isinstance(X, ak.Array):
                # first, get flat numpy view to avoid copy operations
                view = flat_np_view(X)

                # transform into torch Tensor with same type
                values = torch.Tensor(view).to(
                    self.numpy_to_torch_dtype_dict.get(view.dtype.type, view.dtype.type)
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
            X: ak.Array | torch.Tensor | NestedTensor | Mapping[str, ak.Array | torch.Tensor | NestedTensor]
        ) -> (NestedTensor | Mapping[str, NestedTensor]):
            return_tensor: NestedTensor | Mapping[str, NestedTensor] | None = None
            
            if isinstance(X, Mapping):
                return_tensor = {
                    key: self._transform_input(data) for key, data in X.items()
                }
            else:
                return_tensor = self._transform_input(X)

            return return_tensor
        
    class AkToTensor(AkToNestedTensor):
        def _transform_input(self, X):
            return_tensor = None
            if isinstance(X, ak.Array):
                # first, get flat numpy view to avoid copy operations
                try:
                    return_tensor = ak.to_torch(X).to(device=self.device)
                except Exception as e:
                    # default back to NestedTensor
                    print(f"Exception raised in {self.__class__.__name__}: {print(traceback.format_exc())}")
                    return_tensor = super()._transform_input(X)
            elif isinstance(X, (torch.Tensor, NestedTensor)):
                return_tensor = X
            elif isinstance(X, (list, tuple)):
                return_tensor = [self._transform_input(entry) for entry in X]
            elif isinstance(X, dict):
                return_tensor = {key: self._transform_input(val) for key, val in X.items()}

            if return_tensor is None:
                raise ValueError(f"Could not convert input {X=}")
            
            return return_tensor
        
    class PreProcessFloatValues(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.uses = {
                "Tau.{eta,phi,pt,mass,charge}",
                "Electron.{eta,phi,pt,mass,charge}",
                "Muon.{eta,phi,pt,mass,charge}",
                "HHBJet.{pt,eta,phi,mass,hhbtag,btagDeepFlav.*,btagPNet.*}",
                "FatJet.{eta,phi,pt,mass}",
            }
            self.produces = {
                "leptons.*",
                "bjets.*",
                "fatjets.*",
            }

        def rotate_to_phi(self, ref_phi: ak.Array, px: ak.Array, py: ak.Array) -> tuple[ak.Array, ak.Array]:
            """
            Rotates a momentum vector extracted from *events* in the transverse plane to a reference phi
            angle *ref_phi*. Returns the rotated px and py components in a 2-tuple.
            """
            new_phi = np.arctan2(py, px) - ref_phi
            pt = (px**2 + py**2)**0.5
            return pt * np.cos(new_phi), pt * np.sin(new_phi)

        def forward(self, events: ak.Array) -> ak.Array:
            
            # generally attach coffea behavior
            events = attach_coffea_behavior(events, collections={"HHBJet": default_coffea_collections["Jet"]})

            # sanity masks for later usage
            # has_jet_pair = ak.num(events.HHBJet) >= 2
            # has_fatjet = ak.num(events.FatJet) >= 1

            # first extract Leptons
            leptons = attach_behavior(
                ak.concatenate((events.Electron, events.Muon, events.Tau), axis=1),
                type_name="Tau",
            )
            # make sure to actually have two leptons
            has_lepton_pair = ak.num(leptons, axis=1) >= 2
            events = events[has_lepton_pair]
            leptons = leptons[has_lepton_pair]
            lep1, lep2 = leptons[:, 0], leptons[:, 1]

            # calculate phi of lepton system
            phi_lep = np.arctan2(lep1.py + lep2.py, lep1.px + lep2.px)

            def save_rotated_momentum(
                events: ak.Array,
                array: ak.Array,
                target_field: str,
                additional_targets: Collection[str] | None = None
            ) -> ak.Array:
                px, py = self.rotate_to_phi(phi_lep, array.px, array.py)
                # save px and py
                events = set_ak_column(events, f"{target_field}.px", px)
                events = set_ak_column(events, f"{target_field}.py", py)

                routes: set[str] = set(("pz", "energy", "mass"))
                if additional_targets is not None:
                    routes.update(additional_targets)
                for field in routes:
                    events = set_ak_column(events, f"{target_field}.{field}", getattr(array, field))
                return events
            
            events = save_rotated_momentum(events, leptons, "leptons", additional_targets=("charge", "mass"))

            # there might be less than two jets or no fatjet, so pad them
            # bjets = ak.pad_none(_events.HHBJet, 2, axis=1)
            # fatjet = ak.pad_none(_events.FatJet, 1, axis=1)[:, 0]

            jet_columns = ("btagDeepFlavB", "hhbtag", "btagDeepFlavCvB", "btagDeepFlavCvL")
            events = save_rotated_momentum(events, events.HHBJet, "bjets", additional_targets=jet_columns)

            # fatjet variables
            events = save_rotated_momentum(events, events.FatJet, "fatjets", additional_targets=None)
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


