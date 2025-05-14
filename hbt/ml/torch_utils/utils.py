from __future__ import annotations

__all__ = [
    "reorganize_idx", "CustomEarlyStopping", "RelativeEarlyStopping",
]
from collections import defaultdict

import law
from columnflow.util import maybe_import, MockModule
from columnflow.columnar_util import Route, EMPTY_INT, EMPTY_FLOAT
from columnflow.types import Any
from copy import deepcopy


ignite = maybe_import("ignite")
torch = maybe_import("torch")
np = maybe_import("numpy")
ak = maybe_import("awkward")

embedding_expected_inputs = {
    "pair_type": [0, 1, 2],  # see mapping below
    "decay_mode1": [-1, 0, 1, 10, 11],  # -1 for e/mu
    "decay_mode2": [0, 1, 10, 11],
    "lepton1.charge": [-1, 1],
    "lepton2.charge": [-1, 1],
    "has_fatjet": [0, 1],  # whether a selected fatjet is present
    "has_jet_pair": [0, 1],  # whether two or more jets are present
    # 0: 2016APV, 1: 2016, 2: 2017, 3: 2018, 4: 2022preEE, 5: 2022postEE, 6: 2023pre, 7: 2023post
    "year_flag": [0, 1, 2, 3, 4, 5, 6, 7],
    "channel_id": [1, 2, 3],
}
def get_standardization_parameter(
    data_map: list[ParquetDataset],
    columns: list[Route | str] | None = None,
    ) -> dict[str : ak.Array]:
    # open parquet files and concatenate to get statistics for whole datasets
    # beware missing values are currently ignored
    all_data = ak.concatenate(list(map(lambda x: x.data, data_map)))

    statistics = {}
    for _route in columns:
        # ignore empty fields
        arr = _route.apply(all_data)
        # filter missing values out
        empty_mask = arr == EMPTY_FLOAT
        masked_arr = arr[~empty_mask]
        std = ak.std(masked_arr, axis=None)
        mean = ak.mean(masked_arr, axis=None)
        # reshape to 1D array, torch has no interface for 0D
        statistics[_route.column] = {"std": std.reshape(1), "mean": mean.reshape(1)}
    return statistics


def expand_columns(*columns):
    if isinstance(columns, str):
        columns = [columns]

    _columns = set()
    for column_expression in columns:
        if isinstance(column_expression, Route):
            # do nothing if already a route
            break
        expanded_columns = law.util.brace_expand(column_expression)
        routed_columns = set(map(Route, expanded_columns))
        _columns.update(routed_columns)
    return sorted(_columns)


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

    class RelativeEarlyStopping(CustomEarlyStopping):
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


if not any(isinstance(module, MockModule) for module in (torch, np)):
    import torch.nn as nn

    def LookUpTable(array: torch.Tensor, EMPTY=EMPTY_INT, placeholder: int = 15):
        """Maps multiple categories given in *array* into a sparse vectoriced lookuptable.
        Empty values are replaced with *EMPTY*.

        Args:
            array (torch.Tensor): 2D array of categories.
            EMPTY (int, optional): Replacement value if empty. Defaults to columnflow EMPTY_INT.

        Returns:
            tuple([torch.Tensor]): Returns minimum and LookUpTable
        """
        # add placeholder value to array
        array = torch.cat([array, torch.ones(array.shape[0], dtype=torch.int32).reshape(-1, 1) * placeholder], axis=-1)
        # shift input by minimum, pushing the categories to the valid indice space
        minimum = array.min(axis=-1).values
        indice_array = array - minimum.reshape(-1, 1)
        upper_bound = torch.max(indice_array) + 1

        # warn for big categories
        if upper_bound > 100:
            print("Be aware that a large number of categories will result in a large sparse lookup array")

        # create mapping placeholder
        mapping_array = torch.full(
            size=(len(minimum), upper_bound),
            fill_value=EMPTY,
            dtype=torch.int32,
        )

        # fill placeholder with vocabulary

        stride = 0
        # transpose from event to feature loop
        for feature_idx, feature in enumerate(indice_array):
            unique = torch.unique(feature, dim=None)
            mapping_array[feature_idx, unique] = torch.arange(
                stride, stride + len(unique),
                dtype=torch.int32,
            )
            stride += len(unique)
        return minimum, mapping_array

    class CategoricalTokenizer(nn.Module):
        def __init__(self, translation: torch.Tensor, minimum: torch.Tensor):
            """
            This translaytion layer tokenizes categorical features into a sparse representation.
            The input tensor is expected to be a 2D tensor with shape (N, M) where N is the number of events and M is the number of features.
            The output tensor will have shape (N, K) where K is the number of unique categories across all features.

            Args:
                translation (torch.tensor): Sparse representation of the categories, created by LookUpTable.
                minimum (torch.tensor): Array of minimum values for each feature. This is necessary to shift the input tensor to the valid index space.
            """
            super().__init__()
            self.map = translation
            self.min = minimum
            self.indices = torch.arange(len(minimum))

        @property
        def num_dim(self):
            return torch.max(self.map) + 1

        def forward(self, x):
            # shift input array by their respective minimum and slice translation accordingly
            return self.map[self.indices, x - self.min]

        def to(self, *args, **kwargs):
            # make sure to move the translation array to the same device as the input
            self.map = self.map.to(*args, **kwargs)
            self.min = self.min.to(*args, **kwargs)
            self.indices = self.indices.to(*args, **kwargs)
            return super().to(*args, **kwargs)

    def test_(print_comparions=False):
        def prepare_data():

            # unique_values used by the embedding res net
            # type-str, given input, expected output
            unique_values = (
                ("pair_type", [0, 1, 2], [0, 1, 2]),
                ("decay_mode1", [-1, 0, 1, 10, 11], [3, 4, 5, 6, 7]),
                ("decay_mode2", [0, 1, 10, 11], [8, 9, 10, 11]),
                ("charge1", [-1, 1], [12, 13]),
                ("charge2", [-1, 1], [14, 15]),
                ("is_boosted", [0, 1], [16, 17]),
                ("has_jet_pair", [0, 1], [18, 19]),
                ("spin", [0, 2], [20, 21]),
                ("year", [0, 1, 2, 3], [22, 23, 24, 25]),
            )

            data = []
            expected = []
            for name, input, output in unique_values:
                num_cat = len(input)
                input, output = np.array(input), np.array(output)
                # sample events randomly (for small events this will break)
                indices = np.random.choice(num_cat, [1, 10000])
                data.append(input[indices])
                expected.append(output[indices])

            data = np.concatenate(data, axis=0).transpose()
            expected = np.concatenate(expected, axis=0).transpose()
            return (
                torch.from_numpy(data)
                for data in (indices, data, expected)
            )

        indices, input, expected = prepare_data()
        min, translation = LookUpTable(input)
        tokenizer = CategoricalTokenizer(translation, min)

        token = tokenizer(input)
        match_ = token == expected
        assert match_.all(axis=None)
        if print_comparions:
            for ind in range(len(input)):
                print(f"Input: {input[ind]}")
                print(f"Expected: {expected[ind]}")
                print(f"Tokenized: {token[ind]}")
