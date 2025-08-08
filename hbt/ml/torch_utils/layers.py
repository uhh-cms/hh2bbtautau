from __future__ import annotations

__all__ = [
    "PaddingLayer",
    "CategoricalTokenizer",
    "OneHotEncodingLayer",
    "CatEmbeddingLayer",
    "InputLayer",
    "ResNetBlock",
    "DenseBlock",
    "ResNetPreactivationBlock",
    "StandardizeLayer",
    "RotatePhiLayer",
    "AggregationLayer",
    "LBN"
]

from typing import Sequence
import copy

from columnflow.columnar_util import EMPTY_INT, EMPTY_FLOAT, Route
from columnflow.util import maybe_import, MockModule

torch = maybe_import("torch")
PaddingLayer = MockModule("PaddingLayer")  # type: ignore[assignment]
CategoricalTokenizer = MockModule("CategoricalTokenizer")  # type: ignore[assignment]
OneHotEncodingLayer = MockModule("OneHotEncodingLayer")  # type: ignore[assignment]
CatEmbeddingLayer = MockModule("CatEmbeddingLayer")  # type: ignore[assignment]
InputLayer = MockModule("InputLayer")  # type: ignore[assignment]
ResNetBlock = MockModule("ResNetBlock")  # type: ignore[assignment]
DenseBlock = MockModule("DenseBlock")  # type: ignore[assignment]
ResNetPreactivationBlock = MockModule("ResNetPreactivationBlock")  # type: ignore[assignment]
StandardizeLayer = MockModule("StandardizeLayer")  # type: ignore[assignment]
RotatePhiLayer = MockModule("RotatePhiLayer")  # type: ignore[assignment]
AggregationLayer = MockModule("AggregationLayer")  # type: ignore[assignment]
LBN = MockModule("LBN")


if not isinstance(torch, MockModule):
    import torch
    class WeightNormalizedLinear(torch.nn.Linear):  # noqa: F811
        def __init__(self,*args, normalize=True ,**kwargs):
            """
            If normalize is set to True, Linear layer is replaced by weight normalized layer as described in https://arxiv.org/abs/1602.07868.
            If false, the layer is a normal linear layer.

            WeightNormalizedLayer decouple the length of the weight vector from its direction.
            This is done by convert weights from Lienar Layer to weight_original0 and 1.
            0 stands for the magnitude parameter g, while 1 is for the direction v.

            Args:
                normalize (bool, optional): True to replace Linear Layer with WeightNormalizedLayer. Defaults to True.
            """
            super().__init__(*args, **kwargs)
            if normalize:
                self = torch.nn.utils.parametrizations.weight_norm(self, name='weight', dim=0)

    class WeightStandardizationLinear(torch.nn.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(self, *args, **kwargs)

        def forward(self, input):
            weight = self.weight
            weight_mean = weight.mean(dim=-1, keepdim=True)
            std = weight.std(dim=-1 + 1e-5)
            self.weight = (weight - weight_mean)/ std
            return super().forward(input)

    class PaddingLayer(torch.nn.Module):  # noqa: F811
        def __init__(
            self,
            padding_value: float | int = 0,
            mask_value: float | int = EMPTY_FLOAT,
        ):
            """
            Padding layer for torch models. Pads the input tensor with the given padding value.

            Args:
                padding (int, optional): Padding value. Defaults to 0.
            """
            super().__init__()

            self.padding_value = torch.nn.Buffer(torch.tensor(padding_value).to(torch.float32), persistent=True)
            self.mask_value = torch.nn.Buffer(torch.tensor(mask_value).to(torch.float32), persistent=True)

        def forward(self, x):
            mask = x == self.mask_value
            x[mask] = self.padding_value
            return x

    class CategoricalTokenizer(torch.nn.Module):  # noqa: F811
        def __init__(
            self,
            categories: tuple[str | Route],
            expected_categorical_inputs: dict[str, list[int]],
            empty: int = 15,
        ):
            """
            Initializes tokenizer for given *expected_categorical_inputs*.
            The tokenizer creates a mapping array in the order of given columns defined in *categories*.
            Empty values are represented as *empty*.
            All categories will be mapped into a common categorical space and ready to be used by a embedding layer.

            Args:
                categories (tuple[str]): Names of the categories as strings.
                Sorting of the entries must correspond to the order of columns in input tensor!
                expected_categorical_inputs (dict[list[int]], optional): Dictionary where keys are category
                    names and values are lists of integers representing the expected values for
                    each category.
                empty (int, optional): Value used to represent missing values in the input tensor.
                    The empty value must be positive and not already used in the categories.
                    If not given, no handling of missing values will be done.
                    Defaults to 15.
            """
            super().__init__()
            self._expected_inputs, self._empty = self.setup(categories, expected_categorical_inputs, empty)

            # setup lookuptable, returns dummy if None
            _map, _min = self.LookUpTable(self.pad_to_longest())
            _indices = None if _min is None else torch.arange(len(_min))

            # register buffer
            self.map = torch.nn.Buffer(_map, persistent=True)
            self.min = torch.nn.Buffer(_min, persistent=True)
            self.indices = torch.nn.Buffer(_indices, persistent=True)

        def load_state_dict(self, state_dict: dict, strict: bool, assign: bool):
            # overload load_state_dict to set buffer sizes to same of state dict
            for name in self.state_dict().keys():
                self.__setattr__(name, torch.zeros_like(state_dict[name]))
            super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)

        def setup(
            self,
            categories: list[str],
            expected_inputs: list[str],
            empty: int,
        ) -> tuple[dict[str, list[int]], int | None]:
            def _empty(expected_inputs, empty):
                if empty is None:
                    return None
                if empty < 0:
                    raise ValueError("Empty value must be positive")
                if empty in set([item for sublist in expected_inputs.values() for item in sublist]):
                    raise ValueError(f"Empty value {empty} is already used in on the categories")
                return empty

            # check if cateogries are part of expected_inputs at least one
            if not set(categories) & set(expected_inputs.keys()):
                sep = "\n"
                raise ValueError(
                    f"Categories must not be part of Expected categories:\n"
                    f"categories:\n{sep.join(categories)}\nexpected categories:\n{sep.join(expected_inputs.keys())}"
                )

            if expected_inputs is None:
                return {}, None

            # check empty for faulty values
            # add empty category with value of empty to each value
            expected_inputs = copy.deepcopy(expected_inputs)
            empty = _empty(expected_inputs, empty)
            _expected_inputs = {}
            for categorie in map(str, categories):
                data = expected_inputs[categorie]
                data.append(empty)
                _expected_inputs[categorie] = data
            return _expected_inputs, empty

        @property
        def num_dim(self) -> torch.IntTensor:
            return torch.max(self.map) + 1

        def __repr__(self):
            # create dummy input from expected_categorical_inputs
            padded_array = self.pad_to_longest()
            if padded_array is None:
                return "Not initialized Tokenizer"

            expected_pad = padded_array.transpose(0, 1).to(device=self.map.device)
            shifted = expected_pad - self.min
            output_per_feature = self.map[self.indices, shifted].transpose(0, 1)
            _str = []
            _str.append("Translation (input : output):")
            for ind, (categorie, expected_value) in enumerate(self._expected_inputs.items()):
                _str.append(f"{categorie}: {expected_value} -> {output_per_feature[ind][:len(expected_value)].tolist()}")
            return "\n".join(_str)

        def check_for_values_outside_range(self, input_tensor: torch.FloatTensor):
            """
            Helper function checks *input_tensor* for values the tokenizer does not expect but found.

            Args:
                input_tensor (torch.tensor): Input tensor of categorical features.
            """
            # reshape to have features in the first dimension
            input_tensor = input_tensor.transpose(0, 1)
            for i, (categorie, expected_value) in enumerate(self._expected_inputs.items()):
                uniques = set(torch.unique(input_tensor[i]).to(torch.int32).tolist())
                expected = set(expected_value)
                if uniques != expected:
                    difference = uniques - expected
                    print(f"{categorie} has values outside the expected range: {difference}")

        def pad_to_longest(self) -> torch.FloatTensor:
            if not self._expected_inputs:
                return None
            # helper function to pad the input tensor to the longest category
            # first value of the category is used as padding value
            local_max = max([
                len(input_for_categorie)
                for input_for_categorie in self._expected_inputs.values()
            ])
            # pad with first value of the category, so we guarantee to not introduce new values
            array = torch.stack(
                [
                    torch.nn.functional.pad(
                        torch.tensor(input_for_categorie),
                        (0, local_max - len(input_for_categorie)),
                        mode="constant",
                        value=input_for_categorie[0],
                    )
                    for input_for_categorie in self._expected_inputs.values()
                ],
            )
            return array

        def LookUpTable(
            self,
            array: torch.FloatTensor,
            padding_value: int = EMPTY_INT,
        ) -> tuple[torch.FloatTensor, torch.FloatTensor] | None:
            """
            Maps multiple categories given in *array* into a sparse vectoriced lookuptable.
            Empty values are replaced with *EMPTY*.

            Args:
                array (torch.tensor): 2D array of categories.
                EMPTY (int, optional): Replacement value if empty. Defaults to columnflow EMPTY_INT.
                same_empty (bool, optional): If True, all missing values will be mapped to the same value.
                    By default, each category will be mapped to a different value,
                    making it possible to differentia between missing values in different categories.
                    Defaults to False.

            Returns:
                tuple([torch.tensor]), None: Returns minimum and LookUpTable
            """
            if array is None:
                return None, None
            # append empty to the array representing the empty category
            # array = torch.cat([array, torch.ones(array.shape[0], dtype=torch.int32).reshape(-1, 1) * empty], axis=-1)

            # shift input by minimum, pushing the categories to the valid indice space
            minimum = array.min(axis=-1).values
            # shift the input array by their respective minimum
            indice_array = array - minimum.reshape(-1, 1)
            # biggest shifted value + 1
            upper_bound = torch.max(indice_array) + 1

            # warn for big categories
            if upper_bound > 100:
                print("Be aware that a large number of categories will result in a large sparse lookup array")

            # create mapping empty
            mapping_array = torch.full(
                size=(len(minimum), upper_bound),
                fill_value=padding_value,
                dtype=torch.int32,
            )

            # fill empty with vocabulary
            stride = 0
            # transpose from event to feature loop
            for feature_idx, feature in enumerate(indice_array):
                unique = torch.unique(feature, dim=None)
                mapping_array[feature_idx, unique] = torch.arange(
                    stride, stride + len(unique),
                    dtype=torch.int32,
                )
                stride += len(unique)
            return mapping_array, minimum

        def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
            # shift input array by their respective minimum and slice translation accordingly
            # map to int to be used as indices
            shifted = (x - self.min).to(torch.int32)
            output = self.map[self.indices, shifted]
            return output

    class CatEmbeddingLayer(torch.nn.Module):  # noqa: F811
        def __init__(
            self,
            embedding_dim: int,
            categories: tuple[str],
            expected_categorical_inputs: dict[str, list[int]] | None = None,
            category_dims: int | None = None,
            empty: int = 15,
        ):
            """
            Initializes the categorical feature interface with a tokenizer and an embedding layer with
            given *embedding_dim*.

            The tokenizer maps given *categories* to values defined in *expected_categorical_inputs*.
            Missing values are given a *empty* value, which
            The mapping is defined in .
            The embedding layer then maps this combined feature space into a dense representation.

                embedding_dim (int): Number of dimensions for the embedding layer.
                categories (tuple[str]): Names of the categories as strings.
                expected_categorical_inputs (dict[list[int]]): Dictionary where keys are category
                    names and values are lists of integers representing the expected values for
                    each category.
                empty (int, optional): Value used to represent missing values in the input tensor.
            """
            super().__init__()
            self.tokenizer = None
            self.category_dims = category_dims
            if not self.category_dims and all(x is not None for x in (categories, expected_categorical_inputs)):
                self.tokenizer = CategoricalTokenizer(
                    categories=categories,
                    expected_categorical_inputs=expected_categorical_inputs,
                    empty=empty)
                self.category_dims = self.tokenizer.num_dim

            self.embeddings = torch.nn.Embedding(
                self.category_dims,
                embedding_dim,
            )

            self.ndim = embedding_dim * len(categories)

        @property
        def look_up_table(self) -> torch.FloatTensor | None:
            return self.tokenizer.map if self.tokenizer else None

        def normalize_embeddings(self):
            # normalize the embedding layer to have unit length
            with torch.no_grad():
                norm = torch.sqrt(torch.sum(self.embeddings.weight**2, dim=-1)).reshape(-1, 1)
                self.embeddings.weight = torch.nn.Parameter(self.embeddings.weight / norm)

        def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
            if self.tokenizer:
                x = self.tokenizer(x)

            x = self.embeddings(x)
            return x.flatten(start_dim=1)

    class InputLayer(torch.nn.Module):  # noqa: F811
        def __init__(
            self,
            continuous_inputs: tuple[str],
            embedding_dim: int,
            categorical_inputs: tuple[str] | None = None,
            category_dims: int | None = None,
            expected_categorical_inputs: dict[str, list[int]] | None = None,
            empty: int = 15,
            std_layer: torch.nn.Module | None = None,
            rotation_layer: torch.nn.Module | None = None,
            padding_continous_layer: torch.nn.Module | None = None,
            padding_categorical_layer: torch.nn.Module | None = None,
        ):
            """
            Enables the use of categorical and continous features in a single model.
            A tokenizer and embedding layer are created  is created using and an embedding layer.
            The continuous features are passed through a linear layer and then concatenated with the
            categorical features.
            """
            super().__init__()
            self.empty = empty
            self.ndim = len(continuous_inputs)
            self.embedding_layer = None
            if categorical_inputs is not None:
                if expected_categorical_inputs is not None:
                    self.embedding_layer = CatEmbeddingLayer(
                        embedding_dim=embedding_dim,
                        categories=categorical_inputs,
                        expected_categorical_inputs=expected_categorical_inputs,
                        empty=empty)

                elif category_dims:
                    self.embedding_layer = CatEmbeddingLayer(
                        embedding_dim=embedding_dim,
                        category_dims=category_dims,
                        categories=categorical_inputs,
                        empty=empty,
                    )

            if self.embedding_layer:
                self.ndim += self.embedding_layer.ndim

            self.rotation_layer = self.dummy_identity(rotation_layer)
            self.std_layer = self.dummy_identity(std_layer)
            self.padding_continous_layer = self.dummy_identity(padding_continous_layer)
            self.padding_categorical_layer = self.dummy_identity(padding_categorical_layer)

        def dummy_identity(self, layer: torch.nn.Module | None) -> torch.nn.Module:
            if layer is None:
                return torch.nn.Identity()
            return layer

        def cateogrical_preprocessing_pipeline(self, x: torch.FloatTensor) -> torch.FloatTensor:
            x = self.padding_categorical_layer(x)
            return self.embedding_layer(x)

        def continous_preprocessing_pipeline(self, x: torch.FloatTensor) -> torch.FloatTensor:
            # preprocessing
            x = x.to(torch.float32)
            x = self.padding_continous_layer(x)
            x = self.rotation_layer(x)
            x = self.std_layer(x)
            return x

        def forward(self, args):
            categorical_inputs, continuous_inputs = args
            x = torch.cat(
                [
                    self.continous_preprocessing_pipeline(continuous_inputs),
                    self.cateogrical_preprocessing_pipeline(categorical_inputs),
                ],
                dim=1,
            ).to(torch.float32)
            return x

    class ResNetBlock(torch.nn.Module):  # noqa: F811
        def __init__(
            self,
            nodes: int,
            activation_functions: str = "LeakyReLu",
            skip_connection_init: float = 1,
            freeze_skip_connection: float = False,
            eps: float = 1e-5,
            normalize = True,
        ):
            """
            ResNetBlock consisting of a linear layer, batch normalization, and an activation function.
            A adjustable skip connection connects input and output of the block.
            The adjustable skip connection has a learnable parameter, *skip_connection_amplifier*.
            The dimension of the input and output of the block are defined by *nodes*.
            If skip_connection_init is set to 0, the skip connection is disabled.
            This also make if possible to use different in_nodes and out_nodes must can be different.
            To freeze the skip connection parameter, set *freeze_skip_connection* to True.

            Args:
                nodes (int): Number of nodes in the block.
                activation_functions (str, optional): Name of the pytorch activation function, case insenstive.
                    Defaults to "LeakyReLu".
                skip_connection_init (int, optional): Start value of the skipconnection. Defaults to 1.
                freeze_skip_connection (bool, optional): Freeze leanable skipconnection parameter. Defaults to False.
            """
            super().__init__()

            self.nodes = nodes
            self.act_func = self._get_attr(torch.nn.modules.activation, activation_functions)()
            self.skip_connection_amplifier = torch.nn.Parameter(torch.ones(1) * skip_connection_init)
            if freeze_skip_connection:
                self.skip_connection_amplifier.requires_grad = False
            self.layers = torch.nn.Sequential(
                WeightNormalizedLinear(self.nodes, self.nodes, bias=False, normalize=normalize),
                torch.nn.BatchNorm1d(self.nodes, eps=eps),
                self.act_func,
            )

        def _get_attr(self, obj, attr):
            for o in dir(obj):
                if o.lower() == attr.lower():
                    return getattr(obj, o)
            else:
                raise AttributeError(f"Object has no attribute '{attr}'")

        def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
            skip_connection = self.skip_connection_amplifier * x
            x = self.layers(x)
            x = x + skip_connection
            return x

    class DenseBlock(torch.nn.Module):  # noqa: F811
        def __init__(
            self,
            input_nodes: float,
            output_nodes: float,
            activation_functions: str = "LeakyReLu",
            eps: float = 1e-5,
            normalize: bool = True,
        ):
            """
            DenseBlock is a dense block that consists of a linear layer, batch normalization, and an activation function.

            Args:
                nodes (int): Number of nodes in the block.
                activation_functions (str, optional): Name of the pytorch activation function, case insenstive.
                    Defaults to "LeakyReLu".
            """
            super().__init__()
            self.input_nodes = input_nodes
            self.output_nodes = output_nodes

            self.layers = torch.nn.Sequential(
                WeightNormalizedLinear(self.input_nodes, self.output_nodes, bias=False, normalize=normalize),
                torch.nn.BatchNorm1d(self.output_nodes, eps=eps),
                self._get_attr(torch.nn.modules.activation, activation_functions)(),
            )

        def _get_attr(self, obj, attr):
            for o in dir(obj):
                if o.lower() == attr.lower():
                    return getattr(obj, o)
            else:
                raise AttributeError(f"Object has no attribute '{attr}'")

        def forward(self, x):
            return self.layers(x)

    class ResNetPreactivationBlock(torch.nn.Module):  # noqa: F811
        def __init__(
            self,
            nodes: int,
            activation_functions: str = "PReLu",
            skip_connection_init: float = 1,
            freeze_skip_connection: bool = False,
            eps=1e-5,
            normalize: bool = True,
        ):
            """
            Residual block that consists of a linear layer, batch normalization, and an activation function.
            A adjustable skip connection connects input and output of the block.
            The adjustable skip connection has a learnable parameter, *skip_connection_amplifier*.
            The dimension of the input and output of the block are defined by *nodes*.
            If skip_connection_init is set to 0, the skip connection is disabled.
            This also make if possible to use different in_nodes and out_nodes must can be different.
            To freeze the skip connection parameter, set *freeze_skip_connection* to True.

            More information can be found in the original paper: https://arxiv.org/abs/1603.05027

            Args:
                nodes (int): Number of nodes in the block.
                activation_functions (str, optional): Name of the pytorch activation function, case insenstive.
                    Defaults to "LeakyReLu".
                skip_connection_init (int, optional): Start value of the skipconnection. Defaults to 1.
                freeze_skip_connection (bool, optional): Freeze skipconnection parameter. Defaults to False.
            """
            super().__init__()
            self.nodes = nodes
            self.act_func = self._get_attr(torch.nn.modules.activation, activation_functions)()
            self.skip_connection_amplifier = torch.nn.Parameter(torch.ones(1) * skip_connection_init)
            if freeze_skip_connection:
                self.skip_connection_amplifier.requires_grad = False


            self.layers = torch.nn.Sequential(
                WeightNormalizedLinear(self.nodes, self.nodes, bias=False, normalize=normalize),
                torch.nn.BatchNorm1d(self.nodes, eps=eps),
                self.act_func,
                WeightNormalizedLinear(self.nodes, self.nodes, bias=False, normalize=normalize),
                torch.nn.BatchNorm1d(self.nodes, eps=eps),
            )
            self.last_activation = self.act_func

        def _get_attr(self, obj, attr):
            for o in dir(obj):
                if o.lower() == attr.lower():
                    return getattr(obj, o)
            else:
                raise AttributeError(f"Object has no attribute '{attr}'")

        def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
            skip_connection = self.skip_connection_amplifier * x
            x = self.layers(x)
            x = x + skip_connection
            return self.last_activation(x)

    class StandardizeLayer(torch.nn.Module):  # noqa: F811
        def __init__(
            self,
            mean: float = 0.,
            std: float = 1.,
        ):
            """
            Standardizes the input tensor with given *mean* and *std* tensor.
            If no value is provided, mean and std are set to 0 and 1, resulting in no scaling.

            Args:
                mean (torch.tensor, optional): Mean tensor. Defaults to torch.tensor(0.).
                std (torch.tensor, optional): Standard tensor. Defaults to torch.tensor(1.).
            """
            super().__init__()
            self.mean = torch.nn.Buffer(torch.tensor(mean), persistent=True)
            self.std = torch.nn.Buffer(torch.tensor(std), persistent=True)

        def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
            x = (x - self.mean) / self.std
            return x

        def _type_check(self, mean: torch.FloatTensor, std: torch.FloatTensor):
            if not all([isinstance(value, torch.Tensor) for value in [mean, std]]):
                raise TypeError(f"given mean or std needs to be tensor, but is {type(mean)}{type(std)}")

        def update_buffer(self, mean: torch.
                          tensor, std: torch.tensor):
            """
            Update the mean and std parameter.

            Args:
                mean (torch.tensor): Mean value.
                std (torch.tensor): Standard deviation value.
            """
            self._type_check(mean=mean, std=std)
            self.mean = mean.type_as(self.mean)
            self.std = std.type_as(self.std)

    class RotatePhiLayer(torch.nn.Module):  # noqa: F811
        def __init__(
            self,
            columns: list[str] | None,
            ref_phi_columns: list[str] | None = ("lepton1", "lepton2"),
            rotate_columns: list[str] | None = ("bjet1", "bjet2", "fatjet", "lepton1", "lepton2"),
        ):
            """
            Rotate specific *columns* given in *rotate_columns* relative to reference in *ref_phi_columns*.
            """
            super().__init__()
            self.ref_indices = torch.nn.Buffer(self.find_indices_of(columns, ref_phi_columns, True), persistent=True)
            self.rotate_indices = torch.nn.Buffer(self.find_indices_of(columns, rotate_columns, True), persistent=True)

        def load_state_dict(self, state_dict: dict, strict: bool, assign: bool):
            # overload load_state_dict to set buffer sizes to same of state dict
            for name in self.state_dict().keys():
                self.__setattr__(name, torch.zeros_like(state_dict[name]))
            super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)

        def find_indices_of(
            self,
            search_in: list[str],
            search_for: list[str],
            _expand: bool = False,
        ) -> torch.FloatTensor | None:
            if search_in is None or search_for is None:
                return None

            if _expand:
                search_for = self._expand(search_for)
            return torch.tensor([tuple(map(search_in.index, particle)) for particle in search_for])

        def _expand(self, columns: list[str] | str) -> list[tuple[str, str]]:
            # adds px, py to columns and return them as tuple
            columns = [columns] if isinstance(columns, str) else columns
            return [tuple(f"{col}.{suffix}" for suffix in ("px", "py")) for col in columns]

        def calc_phi(
            self,
            x: torch.FloatTensor,
            y: torch.FloatTensor,
        ) -> torch.FloatTensor:
            def arctan2(
                x: torch.FloatTensor,
                y: torch.FloatTensor,
            ) -> torch.FloatTensor:
                # calculate arctan2 using arctan, since onnx does not have support for arctan2
                # torch constants are not tensors for some reason
                pi = torch.tensor(torch.pi)

                # handle special cases
                # quadrant handling: (x < 0,y >= 0) -> arctan + pi, (x < 0,y < 0) -> arctan - pi
                phi_shift = torch.zeros_like(x)
                torch.where(torch.logical_and(x < 0, y < 0), -pi, phi_shift, out=phi_shift)
                torch.where(torch.logical_and(x < 0, y >= 0), pi, phi_shift, out=phi_shift)

                # edges  with x == 0
                phis = torch.arctan(y / x)
                # right edge -> pi/2, left edge -> -pi/2, both 0 -> 0
                torch.where(torch.logical_and(x == 0, y > 0), (pi / 2), phis, out=phis)
                torch.where(torch.logical_and(x == 0, y < 0), -(pi / 2), phis, out=phis)
                torch.where(torch.logical_and(x == 0, y == 0), torch.tensor(0), phis, out=phis)
                return phis + phi_shift
            return arctan2(x=x, y=y)

        def rotate_pt_to_phi(
            self,
            px: torch.FloatTensor,
            py: torch.FloatTensor,
            ref_phi: torch.FloatTensor,
        ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
            # rotate px, py relative to ref_phi
            # returns px, py rotated
            pt = torch.sqrt(torch.square(px) + torch.square(py))
            old_phi = self.calc_phi(x=px, y=py)
            new_phi = old_phi - ref_phi
            return pt * torch.cos(new_phi), pt * torch.sin(new_phi)

        def calc_ref_phi(
            self,
            array,
        ) -> torch.FloatTensor:
            px1, py1 = self.get_kinematics(array, self.ref_indices[0])
            px2, py2 = self.get_kinematics(array, self.ref_indices[1])

            ref_phi = self.calc_phi(
                x=px1 + px2,
                y=py1 + py2,
            )
            return ref_phi

        def get_kinematics(
            self,
            array: torch.FloatTensor,
            ref_indices: torch.FloatTensor,
        ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
            x, y = ref_indices
            return array[:, x], array[:, y]

        def rotate_columns(self, array: torch.FloatTensor, ref_phi: torch.FloatTensor) -> torch.FloatTensor:
            # px, py pairs in fixed order
            for rotate_indice in self.rotate_indices:
                px, py = self.get_kinematics(array, rotate_indice)

                new_px, new_py = self.rotate_pt_to_phi(
                    px=px,
                    py=py,
                    ref_phi=ref_phi,
                )
                array[:, rotate_indice[0]] = new_px
                array[:, rotate_indice[1]] = new_py
            return array

        def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
            ref_phi = self.calc_ref_phi(x)
            return self.rotate_columns(x, ref_phi)

    class TwoHeadFactory(torch.nn.Module):
        def __init__(self, common_base, categorization_head, regression_head):
            """
            Small factory to create a model with two heads.
            All inputs needs to be models of sequential nature, where a single output is expected.

            Args:
                common_base (torch.nn.Module): Network performing common base task
                categorization_head (torch.nn.Module): Network performing categorization task
                regression_head (torch.nn.Module): Network performing regression task
            """
            super().__init__()
            self.base = common_base
            self.cat_head = categorization_head
            self.reg_head = regression_head

        def check_shape(self):
            pass

        def forward(self, x: torch.tensor):
            base_output = self.base(x)
            cat_pred = self.cat_head(base_output)
            reg_pred = self.reg_head(base_output)
            return cat_pred, reg_pred

    class AggregationLayer(torch.nn.Module):
        def __init__(
            self,
            name: str = "sum",
            dim: int = 1,
        ):
            """
            Aggregation Layer to aggregate over object dimension of input.
            This layer assumes that the

            Args:
                name (str, optional): Name of the aggregation function ("max", "sum", "mean"). Defaults to "sum".
                dim (int, optional): Dimension over which the aggregation happens. Defaults to 1.
            """
            super().__init__()
            self.aggregation_fn = self._get_agg_fn(name)
            self.dim = torch.nn.Buffer(torch.tensor(dim).to(torch.int32))

        @property
        def name(self):
            return self.aggregation_fn.__name__

        def _get_agg_fn(self, name):
            return self.__getattribute__(name.lower())

        def max(self, x: torch.FloatTensor) -> torch.FloatTensor:
            return x.max(dim=1).values

        def sum(self, x: torch.FloatTensor) -> torch.FloatTensor:
            return x.sum(dim=1)

        def mean(self, x: torch.FloatTensor) -> torch.FloatTensor:
            return x.mean(dim=1)

        def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
            return self.aggregation_fn(x)

    class LBN(torch.nn.Module):
        """
        Torch implementation of the LBN (Lorentz Boosted Network) feature extractor.
        For details and nomenclature see https://arxiv.org/pdf/1812.09722.
        """

        KNOWN_FEATURES = [
            "e", "px", "py", "pz",
            "pt", "eta", "phi", "m",
            "pair_cos", "pair_dr",
        ]

        DEFAULT_FEATURES = ["e", "pt", "eta", "phi", "m", "pair_cos"]

        def __init__(
            self,
            N: int,
            M: int,
            *,
            features: Sequence[str] | None = None,
            weight_init_scale: float | int = 1.0,
            clip_weights: bool = False,
            eps: float = 1.0e-5,
        ) -> None:
            super().__init__()

            # validate features
            if features is None:
                features = self.DEFAULT_FEATURES
            for f in features:
                if f not in self.KNOWN_FEATURES:
                    raise ValueError(f"unknown feature '{f}', known features are: {self.KNOWN_FEATURES}")

            # store settings
            self.N = N
            self.M = M
            self.features = list(features)
            self.weight_init_scale = weight_init_scale
            self.clip_weights = clip_weights
            self.eps = eps

            # constants
            self.register_buffer("I4", torch.eye(4, dtype=torch.float32))  # (4, 4)
            self.register_buffer("U", torch.tensor([[-1, 0, 0, 0], *(3 * [[0, -1, -1, -1]])], dtype=torch.float32))
            self.register_buffer("U1", self.U + 1)
            self.register_buffer("lower_tril", torch.tril(torch.ones(M, M, dtype=torch.bool), -1))

            # randomly initialized weights for projections
            self.particle_w = torch.nn.Parameter(torch.rand(N, M) * weight_init_scale)
            self.restframe_w = torch.nn.Parameter(torch.rand(N, M) * weight_init_scale)

        def __repr__(self) -> str:
            params = {
                "N": self.N,
                "M": self.M,
                "features": ",".join(self.features),
                "clip": self.clip_weights,
            }
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            return f"{self.__class__.__name__}({params_str}, {hex(id(self))})"

        def update_boosted_vectors(self, boosted_vecs: torch.Tensor) -> torch.Tensor:
            return boosted_vecs

        def forward(self, e: torch.Tensor, px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor) -> torch.Tensor:
            # e, px, py, pz: (B, N)
            E, PX, PY, PZ = range(4)

            # stack 4-vectors
            input_vecs = torch.stack((e, px, py, pz), dim=1)  # (B, 4, N)

            # optionally clip weights to prevent them going negative
            particle_w = self.particle_w
            restframe_w = self.restframe_w
            if self.clip_weights:
                particle_w = torch.clamp(particle_w, min=0.0)
                restframe_w = torch.clamp(restframe_w, min=0.0)

            # create combinations
            particle_vecs = torch.matmul(input_vecs, particle_w)  # (B, 4, M)
            restframe_vecs = torch.matmul(input_vecs, restframe_w)  # (B, 4, M)
            # transpose to (B, M, 4)
            particle_vecs = particle_vecs.permute(0, 2, 1)
            restframe_vecs = restframe_vecs.permute(0, 2, 1)

            # regularize vectors such that e > p
            particle_p = torch.sum(particle_vecs[..., PX:]**2, dim=-1)**0.5  # (B, M)
            particle_vecs[..., E] = torch.maximum(particle_vecs[..., E], particle_p + self.eps)
            restframe_p = torch.sum(restframe_vecs[..., PX:]**2, dim=-1)**0.5  # (B, M)
            restframe_vecs[..., E] = torch.maximum(restframe_vecs[..., E], restframe_p + self.eps)

            # create boost objects
            restframe_m = (restframe_vecs[..., E]**2 - restframe_p**2)**0.5  # (B, M)
            gamma = restframe_vecs[..., E] / restframe_m  # (B, M)
            beta = restframe_p / restframe_vecs[..., E]  # (B, M)
            beta_vecs = restframe_vecs[..., PX:] / restframe_vecs[..., E, None]  # (B, M, 3)
            n_vecs = beta_vecs / beta[..., None]  # (B, M, 3)
            e_vecs = torch.cat([torch.ones_like(n_vecs[..., :1]), -n_vecs], dim=-1)  # (B, M, 4)

            # build Lambda
            Lambda = self.I4 + (
                (self.U + gamma[..., None, None]) *
                (self.U1 * beta[..., None, None] - self.U) *
                (e_vecs[..., None] * e_vecs[..., None, :])
            )  # (B, M, 4, 4)

            # apply boosting
            boosted_vecs = (Lambda @ particle_vecs[..., None])[..., 0]

            # hook to update boosted vectors if desired
            boosted_vecs = self.update_boosted_vectors(boosted_vecs)

            # cached feature provision
            cache = {}
            def get(feature: str) -> torch.Tensor:
                # check cache first
                if feature in cache:
                    return cache[feature]
                # live feature access
                if feature == "e":
                    return boosted_vecs[..., E]
                if feature == "px":
                    return boosted_vecs[..., PX]
                if feature == "py":
                    return boosted_vecs[..., PY]
                if feature == "pz":
                    return boosted_vecs[..., PZ]
                # cached  access
                if feature == "pt2":
                    f = get("px")**2 + get("py")**2
                elif feature == "pt":
                    f = get("pt2")**0.5
                elif feature == "p2":
                    f = get("pt2") + get("pz")**2
                elif feature == "p":
                    f = get("p2")**0.5
                elif feature == "eta":
                    f = torch.atanh(get("pz") / get("p"))
                elif feature == "phi":
                    f = torch.atan2(get("py"), get("px"))
                elif feature == "m":
                    f = (torch.maximum(get("e")**2, get("p2")) - get("p"))**0.5
                elif feature == "pair_cos":
                    boosted_pvecs = boosted_vecs[..., PX:]  # (B, M, 3)
                    boosted_p = get("p")
                    f = (
                        (boosted_pvecs @ boosted_pvecs.transpose(1, 2)) /
                        (boosted_p[..., None] @ boosted_p[:, None, :])
                    )[..., self.lower_tril]  # (B, (M**2-M)/2)
                elif feature == "pair_dr":
                    boosted_phi = get("phi")
                    boosted_eta = get("eta")
                    boosted_dphi = abs(boosted_phi[..., None] - boosted_phi[:, None, :])  # (B, M, M)
                    boosted_dphi = boosted_dphi[..., self.lower_tril]  # (B, (M**2-M)/2)
                    boosted_dphi = torch.where(boosted_dphi > torch.pi, 2 * torch.pi - boosted_dphi, boosted_dphi)
                    boosted_deta = boosted_eta[..., None] - boosted_eta[:, None, :]  # (B, M, M)
                    boosted_deta = boosted_deta[..., self.lower_tril]  # (B, (M**2-M)/2)
                    f = (boosted_dphi**2 + boosted_deta**2)**0.5
                else:
                    raise RuntimeError(f"unknown feature '{feature}'")
                # cache and return
                cache[feature] = f
                return f

            # when not clipping weights, boosted vectors can have e < p
            if not self.clip_weights:
                boosted_vecs[..., E] = torch.maximum(boosted_vecs[..., E], get("p") + self.eps)

            # collect and combine features
            features = torch.cat([get(feature) for feature in self.features], dim=1)  # (B, F)

            return features


    if __name__ == "__main__":
        # hyper-parameters
        N = 10
        M = 5
        bs = 512

        # sample test vectors (simulate numpy's random)
        px = torch.randn(bs, N) * 80.0
        py = torch.randn(bs, N) * 80.0
        pz = torch.randn(bs, N) * 120.0
        m = torch.rand(bs, N) * 50.5 - 0.5
        m = torch.where(m < 0, torch.zeros_like(m), m)
        e = (m**2 + px**2 + py**2 + pz**2)**0.5

        lbn = LBN(N, M, features=LBN.KNOWN_FEATURES)
        feats = lbn(e, px, py, pz)
        print("features shape:", feats.shape)
