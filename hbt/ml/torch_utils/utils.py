from __future__ import annotations

__all__ = [
    "reorganize_idx",
]
from collections import defaultdict
from columnflow.util import maybe_import, MockModule
from columnflow.types import Any
from copy import deepcopy

ignite = maybe_import("ignite")

def reorganize_list_idx(entries):
    first = entries[0]
    if isinstance(first, int):
        return entries
    elif isinstance(first, dict):
        return reorganize_dict_idx(entries)
    elif isinstance(first, (list, tuple)):
        sub_dict = defaultdict(list)
        for e in entries:
            # only the last entry is the idx, all other entries
            # in the list/tuple will be used as keys
            data = e[-1]
            key = tuple(e[:-1])
            if isinstance(data, (list, tuple)):
                sub_dict[key].extend(data)
            else:
                sub_dict[key].append(e[-1])
        return sub_dict

def reorganize_dict_idx(batch):
    return_dict = dict()
    for key, entries in batch.items():
        # type shouldn't change within one set of entries,
        # so just check first
        return_dict[key] = reorganize_list_idx(entries)
    return return_dict

def reorganize_idx(batch):
    if isinstance(batch, dict):
        return reorganize_dict_idx(batch)
    else:
        return reorganize_list_idx(batch)
    
if not isinstance(ignite, MockModule):
    from ignite.handlers import EarlyStopping
    from ignite.engine import Engine
    import torch

    class CustomEarlyStopping(EarlyStopping):
        def __init__(
            self,
            *args,
            model: torch.nn.Module | None = None,
            min_epochs: int = 1,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.best_model: dict[str, Any] | None = None
            self.min_epochs: int = min_epochs
            self.model = model or self.trainer._process_function.keywords["model"]

        def __call__(self, engine: Engine) -> None:
            score = self.score_function(engine)

            if self.best_score is None:
                self.best_score = score
            elif score <= self.best_score + self.min_delta:
                if not self.cumulative_delta and score > self.best_score:
                    self.best_score = score
                self.counter += 1
                self.logger.debug("EarlyStopping: %i / %i" % (self.counter, self.patience))
                if engine.state.epoch > self.min_epochs and self.counter >= self.patience:
                    self.logger.info("EarlyStopping: Stop training")
                    best_epoch = engine.state.epoch - self.patience
                    self.logger.info(f"Resetting model to epoch {getattr(self.trainer, 'best_epoch', best_epoch)}")
                    if self.best_model is not None:
                        self.model.load_state_dict(self.best_model)
                    else:
                        self.logger.warning("No best model found, skipping load")
                    self.trainer.terminate()
            else:
                self.best_score = score
                self.counter = 0
                self.best_model = deepcopy(self.model.state_dict())
                setattr(self.trainer, "best_epoch", engine.state.epoch)