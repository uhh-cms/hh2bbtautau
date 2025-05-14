from __future__ import annotations
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
            placeholder: int = 15,
        ):
            """
            Initializes tokenizer for given *expected_categorical_inputs*.
            The tokenizer creates a mapping array in the order of given columns defined in *categories*.
            Empty values are represented as *placeholder*.
            All given categories will be mapped into a common categorical space and ready to be used by a embedding layer.

            Args:
                categories (tuple[str]): Names of the categories as strings. Sorting of the entries must correspond to the
                    order of columns in input tensor!
                expected_categorical_inputs (dict[list[int]], optional): Dictionary where keys are category
                    names and values are lists of integers representing the expected values for
                    each category.
                placeholder (int, optional): Placeholder value for empty categories.
                    Negative values discouraged, since the tokenizer uses minimal values as shift value.
                    Defaults to 15.
            """
            super().__init__()
            self.placeholder = placeholder
            self.map, self.min = self.LookUpTable(
                self.prepare_mapping(
                    categories=categories,
                    expected_categorical_inputs=expected_categorical_inputs,
                ), placeholder=self.placeholder)

            self.indices = torch.arange(len(self.min))

        @property
        def num_dim(self):
            return torch.max(self.map) + 1

        def print_mapping(self):
            """Prints the mapping of the tokenizer."""
            print("Mapping:")
            for i, categorie in enumerate(self.map):
                print(f"{i}: {categorie}")
            print("Minimum:")
            print(self.min)

        def prepare_mapping(self, categories, expected_categorical_inputs):
            local_max = max([
                len(expected_categorical_inputs[categorie])
                for categorie in categories
            ])

            array = torch.stack(
                [
                    # pad to length of longest category, padding value is the first value of the category
                    nn.functional.pad(
                        torch.tensor(expected_categorical_inputs[categorie]),
                        (0, local_max - len(expected_categorical_inputs[categorie])),
                        mode="constant",
                        value=expected_categorical_inputs[categorie][0],
                    )
                    for categorie in categories
                ],
            )
            return array

        def LookUpTable(self, array: torch.Tensor, EMPTY=EMPTY_INT, placeholder: int = 15):
            """Maps multiple categories given in *array* into a sparse vectoriced lookuptable.
            Empty values are replaced with *EMPTY*.

            Args:
                array (torch.Tensor): 2D array of categories.
                EMPTY (int, optional): Replacement value if empty. Defaults to columnflow EMPTY_INT.

            Returns:
                tuple([torch.Tensor]): Returns minimum and LookUpTable
            """
            # append placeholder to the array representing the empty category
            # array = torch.cat([array, torch.ones(array.shape[0], dtype=torch.int32).reshape(-1, 1) * placeholder], axis=-1)

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
            return mapping_array, minimum

        def forward(self, x):
            # shift input array by their respective minimum and slice translation accordingly
            try:
                output = self.map[self.indices, x - self.min]
            except Exception as e:
                print(f"Error in CategoricalTokenizer: {e}")
                print(f"Input: {x}")
                print(f"Min: {self.min}")
                print(f"Indices: {self.indices}")
                print(f"Map: {self.map}")
                from IPython import embed
                embed(header=f"Error in CategoricalTokenizer: {e}")
                raise e
            return output

        def to(self, *args, **kwargs):
            # make sure to move the translation array to the same device as the input
            self.map = self.map.to(*args, **kwargs)
            self.min = self.min.to(*args, **kwargs)
            self.indices = self.indices.to(*args, **kwargs)
            return super().to(*args, **kwargs)


    class CatEmbeddingLayer(nn.Module):
        def __init__(
            self,
            embedding_dim: int,
            categories: tuple[str],
            expected_categorical_inputs: dict[list[int]],
            placeholder: int = 15,
        ):
            """
            Initializes the categorical feature interface with a tokenizer and an embedding layer with
            given *embedding_dim*.

            The tokenizer maps given *categories* to values defined in *expected_categorical_inputs*.
            Missing values are given a *placeholder* value, which
            The mapping is defined in .
            The embedding layer then maps this combined feature space into a dense representation.

                embedding_dim (int): Number of dimensions for the embedding layer.
                categories (tuple[str]): Names of the categories as strings.
                expected_categorical_inputs (dict[list[int]]): Dictionary where keys are category
                    names and values are lists of integers representing the expected values for
                    each category.
            """
            super().__init__()

            self.tokenizer = CategoricalTokenizer(
                categories=categories,
                expected_categorical_inputs=expected_categorical_inputs,
                placeholder=placeholder)

            self.embeddings = torch.nn.Embedding(
                self.tokenizer.num_dim,
                embedding_dim,
            )

            self.final_embeddings = torch.nn.Sequential(
                # self.embeddings,
                nn.Flatten(),
                nn.BatchNorm1d(embedding_dim*len(categories)),
                # nn.Linear(embedding_dim*len(categories), 10),
                # nn.ReLU(),
            )

            self.ndim = 10

        @property
        def look_up_table(self):
            return self.tokenizer.map

        def forward(self, cat_input):
            x = self.tokenizer(cat_input)
            x = self.embeddings(x)
            x = self.final_embeddings(x)
            return x.flatten(start_dim=1)

        def to(self, *args, **kwargs):
            self.tokenizer.to(*args, **kwargs)
            return super().to(*args, **kwargs)

    class InputLayer(nn.Module):
        def __init__(
            self,
            continuous_inputs: tuple[str],
            categorical_inputs: tuple[str],
            embedding_dim: int,
            expected_categorical_inputs: dict[list[int]],
            placeholder: int = 15,
        ):
            """
            Enables the use of categorical and continous features in a single model.
            A tokenizer and embedding layer are created  is created using and an embedding layer.
            The continuous features are passed through a linear layer and then concatenated with the
            categorical features.
            """
            super().__init__()
            self.placeholder = placeholder
            self.ndim = len(continuous_inputs)
            if categorical_inputs is not None and expected_categorical_inputs is not None:
                self.embedding_layer = CatEmbeddingLayer(
                    embedding_dim=embedding_dim,
                    categories=categorical_inputs,
                    expected_categorical_inputs=expected_categorical_inputs,
                    placeholder=placeholder)

                self.ndim += embedding_dim * len(categorical_inputs)

        def forward(self, continuous_inputs, categorical_inputs):
            x = torch.cat(
                [
                    continuous_inputs,
                    self.embedding_layer(categorical_inputs),
                ],
                dim=1,
            )

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
            self.mean = self.mean.to(*args, **kwargs)
            self.std = self.std.to(*args, **kwargs)
            return super().to(*args, **kwargs)
