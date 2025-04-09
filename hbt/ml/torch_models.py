from __future__ import annotations

__all__ = [
    "model_clss",
]

from functools import partial

from collections import Iterable
from collections.abc import Container, Collection
from columnflow.util import MockModule, maybe_import, DotDict
from columnflow.types import T, Any, Callable, Sequence
from columnflow.columnar_util import Route, EMPTY_FLOAT, EMPTY_INT

torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
np = maybe_import("numpy")
ak = maybe_import("awkward")
import law

model_clss: DotDict[str, torch.nn.Module] = DotDict()

if not isinstance(torch, MockModule):
    from torch import nn
    from torch.optim import Adam
    import torchdata.nodes as tn

    from hbt.ml.torch_utils.datasets import FlatRowgroupParquetDataset, FlatArrowRowGroupParquetDataset
    from hbt.ml.torch_utils.transforms import AkToTensor, PreProcessFloatValues
    from hbt.ml.torch_utils.samplers import ListRowgroupSampler
    from hbt.ml.torch_utils.map_and_collate import NestedListRowgroupMapAndCollate
    from hbt.ml.torch_utils.dataloaders import CompositeDataLoader
    from hbt.ml.torch_utils.datasets.handlers import FlatListRowgroupParquetFileHandler, FlatArrowParquetFileHandler

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
                nn.Sigmoid(),
            )

            self.loss_fn = nn.BCELoss()

        def init_optimizer(self, learning_rate=1e-3, weight_decay=1e-5):
            return Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        def _build_categorical_target(self, dataset: str):
            return int(1) if dataset.startswith("hh") else int(0)
        
        def init_dataset_handler(self, task: law.Task):
            group_datasets = {
                "ttbar": [d for d in task.datasets if d.startswith("tt_")]
            }
            device = next(self.parameters()).device
            self.dataset_handler = FlatListRowgroupParquetFileHandler(
                task=task,
                columns=self.inputs,
                batch_transformations=AkToTensor(device=device),
                global_transformations=PreProcessFloatValues(),
                build_categorical_target_fn=self._build_categorical_target,
                group_datasets=group_datasets,
                device=device,
            )

        def init_datasets(self):
            return self.dataset_handler.init_datasets()
        
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
        
    class DropoutFeedForwardNet(FeedForwardNet):
        def __init__(self):
            super().__init__()

            self.linear_relu_stack = nn.Sequential(
                nn.BatchNorm1d(len(self.inputs),),
                # nn.Dropout(p=0.2),
                nn.Linear(len(self.inputs), 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                # nn.Dropout(p=0.2),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                # nn.Dropout(p=0.2),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
                nn.Sigmoid(),
            )

    class FeedForwardArrow(FeedForwardNet):
        def init_dataset_handler(self, task: law.Task):
            group_datasets = {
                "ttbar": [d for d in task.datasets if d.startswith("tt_")]
            }
            device = next(self.parameters()).device
            self.dataset_handler = FlatArrowParquetFileHandler(
                task=task,
                columns=self.inputs,
                batch_transformations=AkToTensor(device=device),
                global_transformations=PreProcessFloatValues(),
                build_categorical_target_fn=self._build_categorical_target,
                group_datasets=group_datasets,
                device=device,
            )

    model_clss["feedforward"] = FeedForwardNet
    model_clss["feedforward_arrow"] = FeedForwardArrow

    model_clss["feedforward_dropout"] = DropoutFeedForwardNet

