from columnflow.columnar_util import EMPTY_INT

import torch.nn as nn
import torch


class CategoricalTokenizer_New(nn.Module):
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
            categories (tuple[str]): Names of the categories as strings.
            expected_categorical_inputs (dict[list[int]], optional): Dictionary where keys are category
                names and values are lists of integers representing the expected values for
                each category.
            placeholder (int, optional): Placeholder value for empty categories.
                Negative values discouraged, since the tokenizer uses minimal values as shift value.
                Defaults to 15.
        """
        super().__init__()
        self.map, self.min = self.LookUpTable(
            self.prepare_mapping(
                categories=categories,
                expected_categorical_inputs=expected_categorical_inputs,
            ), placeholder=placeholder)

        self.indices = torch.arange(len(self.min))

    @property
    def num_dim(self):
        return torch.max(self.map) + 1

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
        return mapping_array, minimum

    def forward(self, x):
        # shift input array by their respective minimum and slice translation accordingly
        return self.map[self.indices, x - self.min]

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
        Initializes the categorical feature interface with a tokenizer and an embedding layer.

        The tokenizer maps categorical features into a combined feature space and includes an
        empty category for missing values. The embedding layer then maps this combined feature
        space into a dense representation.

            embedding_dim (int): Number of dimensions for the embedding layer.
            categories (tuple[str]): Names of the categories as strings.
            expected_categorical_inputs (dict[list[int]]): Dictionary where keys are category
                names and values are lists of integers representing the expected values for
                each category.
        """
        super().__init__()
        self.tokenizer = CategoricalTokenizer_New(
            categories=categories,
            expected_categorical_inputs=expected_categorical_inputs,
            placeholder=placeholder)

        self.embeddings = torch.nn.Embedding(
            self.tokenizer.num_dim,
            embedding_dim,
        )

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.embeddings(x)
        return x.flatten(start_dim=1)
