from __future__ import annotations
import copy

from columnflow.columnar_util import EMPTY_INT, EMPTY_FLOAT
from columnflow.util import maybe_import, MockModule

torch = maybe_import("torch")

if not isinstance(torch, MockModule):
    from torch import nn

    class PaddingLayer(nn.Module):
        def __init__(self, padding_value: float | int = 0, mask_value: float | int = EMPTY_FLOAT):
            """
            Padding layer for torch models. Pads the input tensor with the given padding value.

            Args:
                padding (int, optional): Padding value. Defaults to 0.
            """
            super().__init__()
            self.padding_value = padding_value
            self.mask_value = mask_value

        def forward(self, x):
            mask = x == self.mask_value
            x[mask] = self.padding_value
            return x

    class CategoricalTokenizer(nn.Module):
        def __init__(
            self,
            categories: tuple[str],
            expected_categorical_inputs: dict[list[int]],
            empty: int = 15,
        ):
            """
            Initializes tokenizer for given *expected_categorical_inputs*.
            The tokenizer creates a mapping array in the order of given columns defined in *categories*.
            Empty values are represented as *empty*.
            All given categories will be mapped into a common categorical space and ready to be used by a embedding layer.

            Args:
                categories (tuple[str]): Names of the categories as strings. Sorting of the entries must correspond to the
                    order of columns in input tensor!
                expected_categorical_inputs (dict[list[int]], optional): Dictionary where keys are category
                    names and values are lists of integers representing the expected values for
                    each category.
                empty (int, optional): Value used to represent missing values in the input tensor.
                    The empty value must be positive and not already used in the categories.
                    If not given, no handling of missing values will be done.
                    Defaults to 15.
            """
            super().__init__()
            self.expected_categorical_inputs = copy.deepcopy(expected_categorical_inputs)
            self.categories = categories
            self.empty = self._empty(empty)

            self.map, self.min = self.LookUpTable(
                self.pad_to_longest())

            self.indices = torch.arange(len(self.min))

        @property
        def num_dim(self):
            return torch.max(self.map) + 1

        def _empty(self, empty):
            if empty is None:
                return None

            if empty < 0:
                raise ValueError("Empty value must be positive")
            if empty in set([item for sublist in self.expected_categorical_inputs.values() for item in sublist]):
                raise ValueError(f"Empty value {empty} is already used in on the categories")

            # add empty to the expected_categorical_inputs
            for categorie in self.categories:
                self.expected_categorical_inputs[categorie].append(empty)
            return empty

        def __repr__(self):
            # create dummy input from expected_categorical_inputs
            expected_pad = self.pad_to_longest().transpose(0, 1).to(device=self.map.device)
            shifted = expected_pad - self.min
            output_per_feature = self.map[self.indices, shifted].transpose(0, 1)
            _str = []
            _str.append("Translation (input : output):")
            for ind, categorie in enumerate(self.categories):
                num_expected = self.expected_categorical_inputs[categorie]
                _str.append(f"{categorie}: {num_expected} -> {output_per_feature[ind][:len(num_expected)].tolist()}")
            return "\n".join(_str)

        def check_for_values_outside_range(self, input_tensor):
            """
            Helper function checks *input_tensor* for values the tokenizer does not expect but found.

            Args:
                input_tensor (torch.tensor): Input tensor of categorical features.
            """
            from IPython import embed; embed(header="check - 99 in layers.py ")
            # reshape to have features in the first dimension
            input_tensor = input_tensor.transpose(0,1)
            for i, categorie in enumerate(self.categories):
                uniques = set(torch.unique(input_tensor[i]).to(torch.int32).tolist())
                expected = set(self.expected_categorical_inputs[categorie])
                if uniques != expected:
                    difference = uniques - expected
                    print(f"{categorie} has values outside the expected range: {difference}")

        def pad_to_longest(self):
            # helper function to pad the input tensor to the longest category, using the first value of the category as padding value
            local_max = max([
                len(self.expected_categorical_inputs[categorie])
                for categorie in self.categories
            ])
            # pad with first value of the category, so we guarantee to not introduce new values
            array = torch.stack(
                [
                    nn.functional.pad(
                        torch.tensor(self.expected_categorical_inputs[categorie]),
                        (0, local_max - len(self.expected_categorical_inputs[categorie])),
                        mode="constant",
                        value=self.expected_categorical_inputs[categorie][0],
                    )
                    for categorie in self.categories
                ],
            )
            return array

        def LookUpTable(self, array: torch.Tensor, padding_value=EMPTY_INT):
            """
            Maps multiple categories given in *array* into a sparse vectoriced lookuptable.
            Empty values are replaced with *EMPTY*.

            Args:
                array (torch.Tensor): 2D array of categories.
                EMPTY (int, optional): Replacement value if empty. Defaults to columnflow EMPTY_INT.
                same_empty (bool, optional): If True, all missing values will be mapped to the same value.
                    By default, each category will be mapped to a different value,
                    making it possible to differentia between missing values in different categories.
                    Defaults to False.

            Returns:
                tuple([torch.Tensor]): Returns minimum and LookUpTable
            """
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

        def forward(self, x):
            # shift input array by their respective minimum and slice translation accordingly
            output = self.map[self.indices, x - self.min]
            return output

        def to(self, *args, **kwargs):
            # make sure to move the translation array to the same device as the input
            self.map = self.map.to(*args, **kwargs)
            self.min = self.min.to(*args, **kwargs)
            self.indices = self.indices.to(*args, **kwargs)
            return super().to(*args, **kwargs)

    class OneHotEncodingLayer(nn.Module):
        def __init__(self, num_classes):
            """
            One hot encoding layer for torch models. One hot encodes the input tensor with the given padding value.
            """
            super().__init__()
            self.one_hot = nn.functional.one_hot
            self.num_classes = num_classes

        def forward(self, x):
            return self.one_hot(x, num_classes=self.num_classes).int()

        def to(self, *args, **kwargs):
            self.one_hot = self.one_hot.to(*args, **kwargs)
            return super().to(*args, **kwargs)


    class CatEmbeddingLayer(nn.Module):
        def __init__(
            self,
            embedding_dim: int,
            categories: tuple[str],
            expected_categorical_inputs: dict[list[int]] | None = None,
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


            self.ndim = embedding_dim*len(categories)

        @property
        def look_up_table(self):
            return self.tokenizer.map if self.tokenizer else None

        def normalize_embeddings(self):
            # normalize the embedding layer to have unit length
            with torch.no_grad():
                norm = torch.sqrt(torch.sum(self.embeddings.weight**2, dim=-1)).reshape(-1,1)
                self.embeddings.weight = torch.nn.Parameter(self.embeddings.weight / norm)

        def forward(self, cat_input):
            x = cat_input

            if self.tokenizer:
                x = self.tokenizer(x)

            x = self.embeddings(x)
            return x.flatten(start_dim=1)

        def to(self, *args, **kwargs):
            if self.tokenizer:
                self.tokenizer.to(*args, **kwargs)
            return super().to(*args, **kwargs)

    class InputLayer(nn.Module):
        def __init__(
            self,
            continuous_inputs: tuple[str],
            embedding_dim: int,
            categorical_inputs: tuple[str] = None,
            category_dims: int | None = None,
            expected_categorical_inputs: dict[list[int]] | None = None,
            empty: int = 15,
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

        def forward(self, args):
            categorical_inputs, continuous_inputs = args
            x = torch.cat(
                [
                    continuous_inputs,
                    self.embedding_layer(categorical_inputs),
                ],
                dim=1,
            )
            return x

        def to(self, *args, **kwargs):
            self.embedding_layer.to(*args, **kwargs)
            return super().to(*args, **kwargs)

    class ResNetBlock(nn.Module):
        def __init__(
            self,
            nodes,
            activation_functions="LeakyReLu",
            skip_connection_init=1,
            freeze_skip_connection=False,
        ):
            """
            ResNetBlock is a residual block that consists of a linear layer, batch normalization, and an activation function.
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
                freeze_skip_connection (bool, optional): Turn off learning for skipconnection parameter. Defaults to False.
            """
            super().__init__()
            self.nodes = nodes
            self.act_func = self._get_attr(nn.modules.activation, activation_functions)()
            self.skip_connection_amplifier = nn.Parameter(torch.ones(1) * skip_connection_init)
            if freeze_skip_connection:
                self.skip_connection_amplifier.requires_grad = False
            self.layers = nn.Sequential(
                nn.Linear(self.nodes, self.nodes, bias=False),
                nn.BatchNorm1d(self.nodes),
                self.act_func,
            )

        def _get_attr(self, obj, attr):
            for o in dir(obj):
                if o.lower() == attr.lower():
                    return getattr(obj, o)
            else:
                raise AttributeError(f"Object has no attribute '{attr}'")

        def forward(self, x):
            skip_connection = self.skip_connection_amplifier * x
            x = self.layers(x)
            x = x + skip_connection
            return x

    class StandardizeLayer(nn.Module):
        def __init__(self, mean: float | int = 0, std: float | int = 1):
            """
            Standardize layer for torch models. Standardizes the input tensor with the given mean and std.

            Args:
                mean (float, optional): Mean value. Defaults to 0.
                std (float, optional): Standard deviation value. Defaults to 1.
            """
            super().__init__()
            self.mean = mean
            self.std = std

        def forward(self, x):
            x = (x - self.mean) / self.std
            return x

        def set_mean_std(self, mean: float | int = 0, std: float | int = 1):
            """
            Set the mean and std values for the standardization layer.

            Args:
                mean (float, optional): Mean value. Defaults to 0.
                std (float, optional): Standard deviation value. Defaults to 1.
            """
            self.mean = mean
            self.std = std

        def to(self, *args, **kwargs):
            if isinstance(self.mean, torch.Tensor):
                self.mean = self.mean.to(*args, **kwargs)
            if isinstance(self.std, torch.Tensor):
                self.std = self.std.to(*args, **kwargs)
            return super().to(*args, **kwargs)
