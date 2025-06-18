from __future__ import annotations

__all__ = [
    "reorganize_idx", "CustomEarlyStopping", "RelativeEarlyStopping",
]
from collections import defaultdict
from collections.abc import Iterable

import law
from columnflow.util import maybe_import, MockModule
from columnflow.columnar_util import Route, EMPTY_INT, EMPTY_FLOAT
from columnflow.types import Any, Callable
from copy import deepcopy
from hbt.ml.torch_utils.datasets import ParquetDataset

ignite = maybe_import("ignite")
torch = maybe_import("torch")
np = maybe_import("numpy")
ak = maybe_import("awkward")  # type: ignore
onnx = maybe_import("onnx")
rt = maybe_import("onnxruntime")


embedding_expected_inputs = {
    "pair_type": [0, 1, 2],  # see mapping below
    "decay_mode1": [-1, 0, 1, 10, 11],  # -1 for e/mu
    "decay_mode2": [0, 1, 10, 11],
    "lepton1.charge": [-1, 1, 0],
    "lepton2.charge": [-1, 1, 0],
    "has_fatjet": [0, 1],  # whether a selected fatjet is present
    "has_jet_pair": [0, 1],  # whether two or more jets are present
    # 0: 2016APV, 1: 2016, 2: 2017, 3: 2018, 4: 2022preEE, 5: 2022postEE, 6: 2023pre, 7: 2023post
    "year_flag": [0, 1, 2, 3, 4, 5, 6, 7],
    "channel_id": [1, 2, 3],
}


def export_ensemble_onnx(
    ensemble_wrapper,
    categoricat_tensor: torch.Tensor,
    continous_tensor: torch.Tensor,
    save_dir: str,
    opset_version: int = None,
) -> str:
    """
    Wrapper to export an ensemble model to onnx format. Does the same as export_onnx, but
    iterates over all models in the ensemble_wrapper and freezes them before exporting.

    Args:
        ensemble_wrapper (MLEnsembleWrapper): _description_
        categoricat_tensor (torch.tensor): tensor representing categorical features
        continous_tensor (torch.tensor): tensor representing categorical features
        save_dir (str): directory where the onnx model will be saved.
        opset_version (int, optional): version of the used operation sets. Defaults to None.

    Returns:
        str: The path of the saved onnx ensemble model.
    """
    for model in ensemble_wrapper.models:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    export_onnx(
        ensemble_wrapper,
        categoricat_tensor,
        continous_tensor,
        save_dir,
        opset_version=opset_version,
    )


def export_onnx(
    model: torch.nn.Module,
    categoricat_tensor: torch.Tensor,
    continous_tensor: torch.Tensor,
    save_dir: str,
    opset_version: int = None,
) -> str:
    """
    Function to export a loaded pytorch *model* to onnx format saved in *save_dir*.
    To successfully export the model, the input tensors *categoricat_tensor* and *continous_tensor* must be provided.
    For backwards compatibility, an opset_version can be enforced.
    A table about which opsets are available can be found here: https://onnxruntime.ai/docs/reference/compatibility.html
    Some operations are only available in newer opsets, or change behavior inbetween version.
    A list of all operations and their versions is given at: https://onnx.ai/onnx/operators/

    Args:
        model (torch.nn.model): loaded Pytorch model, ready to perform inference.
        categoricat_tensor (torch.tensor): tensor representing categorical features
        continous_tensor (torch.tensor): tensor representing categorical features
        save_dir (str): directory where the onnx model will be saved.
        opset_version (int, optional): version of the used operation sets. Defaults to None.

    Returns:
        str: The path of the saved onnx model.
    """

    logger = law.logger.get_logger(__name__)

    onnx_version = onnx.__version__
    runtime_version = rt.__version__
    torch_version = torch.__version__

    save_path = f"{save_dir}-onnx_{onnx_version}-rt_{runtime_version}-torch{torch_version}.onnx"

    # prepare export
    num_cat_features = categoricat_tensor.shape[-1]
    num_cont_features = continous_tensor.shape[-1]

    # cast to proper format, numpy and float32
    categoricat_tensor = categoricat_tensor.numpy().astype(np.float32).reshape(-1, num_cat_features)
    continous_tensor = continous_tensor.numpy().astype(np.float32).reshape(-1, num_cont_features)

    # double bracket is necessary since onnx, and our model unpacks the input tuple
    input_feed = ((categoricat_tensor, continous_tensor),)

    torch.onnx.export(
        model,
        input_feed,
        save_path,
        input_names=["cat", "cont"],
        output_names=["output"],
        # if opset is none highest available will be used

        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes={
            # enable dynamic batch sizes
            "cat": {0: "batch_size"},
            "cont": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    logger.info(f"Succefully exported onnx model to {save_path}")
    return save_path


def test_run_onnx(
    model_path: str,
    categorical_array: np.ndarray,
    continous_array: np.ndarray,
) -> np.ndarray:
    """
    Function to run a test inference on a given *model_path*.
    The *categorical_array* and *continous_array* are expected to be given as numpy arrays.

    Args:
        model_path (str): Model path to onnx model
        categorical_array (np.ndarray): Array of categorical features
        continous_array (np.ndarray): Array of continous features

    Returns:
        np.ndarray: Prediction of the model
    """
    sess = rt.InferenceSession(model_path, providers=rt.get_available_providers())
    first_node = sess.get_inputs()[0]
    second_node = sess.get_inputs()[1]

    # setup data
    input_feed = {
        first_node.name: categorical_array.reshape(-1, first_node.shape).astype(np.float32),
        second_node.name: continous_array.reshape(-1, second_node.shape).astype(np.float32),
    }

    output_name = [output.name for output in sess.get_outputs()]

    onnx_predition = sess.run(output_name, input_feed)
    return onnx_predition


def get_standardization_parameter(
    data_map: list[ParquetDataset],
    columns: Iterable[Route | str],
) -> dict[str, ak.Array]:
    # open parquet files and concatenate to get statistics for whole datasets
    # beware missing values are currently ignored
    all_data = ak.concatenate(list(map(lambda x: x.data, data_map)))
    # make sure columns are Routes
    columns = list(map(Route, columns))

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


CustomEarlyStopping = MockModule("CustomEarlyStopping")  # type: ignore

if not isinstance(ignite, MockModule):
    from ignite.handlers import EarlyStopping
    from ignite.engine import Engine
    import torch

    class CustomEarlyStopping(EarlyStopping):  # noqa: F811
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
            self.epoch = 0

        def __call__(self, engine: Engine) -> None:
            score = self.score_function(engine)
            self.epoch += 1

            if self.best_score is None:
                self.logger.debug(f"{self.__class__.__name__}: Initializing best score with {score}")
                self.best_score = score
            elif score <= self.best_score + self.min_delta:
                if not self.cumulative_delta and score > self.best_score:
                    self.best_score = score
                self.counter += 1
                self.logger.info("EarlyStopping: %i / %i" % (self.counter, self.patience))

                if self.epoch > self.min_epochs and self.counter >= self.patience:
                    self.logger.info("EarlyStopping: Stop training")
                    best_epoch = self.epoch - self.patience
                    self.logger.info(f"Resetting model to epoch {getattr(self.trainer, 'best_epoch', best_epoch)}")
                    if self.best_model is not None:
                        self.model.load_state_dict(self.best_model)
                    else:
                        self.logger.warning("No best model found, skipping load")
                    self.trainer.terminate()
            else:
                self.logger.debug(f"{self.__class__.__name__}: Previous best score {self.best_score}, {self.min_delta=}")
                self.logger.debug(f"{self.__class__.__name__}: setting best score to {score}")
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
    from torch import Tensor

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
            The input tensor is a 2D tensor with shape (N, M) where N is the number of events and M of features.
            The output tensor will have shape (N, K) where K is the number of unique categories across all features.

            Args:
                translation (torch.tensor): Sparse representation of the categories, created by LookUpTable.
                minimum (torch.tensor): Array of minimas used to shift the input tensor into the valid index space.
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

    class MLEnsembleWrapper(nn.Module):
        def __init__(self, models: Iterable[nn.Module] | nn.Module, final_activation: Callable | str = "softmax"):
            """
            This wrapper allows to use a model that expects a sparse representation of categorical features.
            The input tensor is a 2D tensor with shape (N, M) where N is the number of events and M of features.
            The output tensor will have shape (N, K) where K is the number of unique categories across all features.

            Args:
                model Iterable[nn.Module] | nn.Module: Model or models that need to be evaluated.
            """
            super().__init__()

            self.models: Iterable[nn.Module] = models if isinstance(models, Iterable) else [models]
            self.final_activation: Callable
            if callable(final_activation):
                self.final_activation = final_activation
            elif isinstance(final_activation, str) and hasattr(nn.functional, final_activation):
                self.final_activation = getattr(nn.functional, final_activation)
            else:
                raise ValueError(f"Invalid final activation function: {final_activation}")

        def forward(self, x):
            outputs: list[Tensor] = []
            for model in self.models:
                outputs.append(self.final_activation(model(*x)))

            # collect outputs in tensor for easier handling
            output_tensor = torch.cat([o[..., None] for o in outputs], axis=-1)
            return torch.mean(output_tensor, axis=-1), torch.std(output_tensor, axis=-1)
