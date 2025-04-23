import unittest
import os

from hbt.ml.torch_utils.batcher import BatchedMultiNodeWeightedSampler

this_dir = os.path.realpath(os.path.dirname(__file__))


class TorchUtilTests(unittest.TestCase):

    def setUp(self):
        from hbt.ml.torch_utils.datasets import ListDataset
        from hbt.ml.torch_utils.dataloaders import NodesDataLoader
        from torch.utils.data import default_collate
        import json

        # load the results
        with open(os.path.join(this_dir, "torch_util_results_balanced.json"), "r") as f:
            self.balanced_results = json.load(f)

        self.data_s = ListDataset(15, "signal")
        self.data_b = ListDataset(43, "background")
        self.data_b2 = ListDataset(25, "bkg2")
        self.node_s = NodesDataLoader(
            self.data_s,
            shuffle=False,
            num_workers=1,
            collate_fn=default_collate,
            pin_memory=False,
        )

        self.node_b = NodesDataLoader(
            self.data_b,
            shuffle=False,
            num_workers=1,
            collate_fn=default_collate,
            pin_memory=False,
        )

        self.node_b2 = NodesDataLoader(
            self.data_b2,
            shuffle=False,
            num_workers=1,
            collate_fn=default_collate,
            pin_memory=False,
        )

    def _compare_values(self, node_dict, weight_dict, results):

        for i, result in results.items():
            with self.subTest(i=i):

                # if i is a str, raise an error
                with self.assertRaises(TypeError):
                    composite_batched_sampler = BatchedMultiNodeWeightedSampler(
                        node_dict,
                        weights=weight_dict,
                        batch_size=i,
                    )
                # should work when casting to int though
                composite_batched_sampler = BatchedMultiNodeWeightedSampler(
                    node_dict,
                    weights=weight_dict,
                    batch_size=int(i),
                )
                self.assertListEqual(
                    [list(list(y) for y in x) for x in composite_batched_sampler],
                    result,
                )

    def test_BatchedMultiNodeWeightedSampler_balanced(self):
        import torchdata.nodes as tn
        import json

        # load the results
        with open(os.path.join(this_dir, "torch_util_results_balanced.json"), "r") as f:
            results = json.load(f)

        node_dict = {"signal": tn.SamplerWrapper(self.node_s), "bkg": tn.SamplerWrapper(self.node_b)}

        self._compare_values(node_dict, {"signal": 1., "bkg": 1.}, results)

    def test_BatchedMultiNodeWeightedSampler_unbalanced(self):
        import torchdata.nodes as tn
        import json

        # load the results
        with open(os.path.join(this_dir, "torch_util_results_unbalanced.json"), "r") as f:
            results = json.load(f)

        node_dict = {"signal": tn.SamplerWrapper(self.node_s), "bkg": tn.SamplerWrapper(self.node_b)}

        self._compare_values(node_dict, {"signal": 2., "bkg": 1.}, results)

    def test_BatchedMultiNodeWeightedSampler_subsampling(self):
        import torchdata.nodes as tn
        from hbt.ml.torch_utils.batcher import BatchedMultiNodeWeightedSampler

        # load the results
        # import json
        # with open(os.path.join(this_dir, "torch_util_results_unbalanced.json"), "r") as f:
        #     results = json.load(f)

        node_dict = {
            "signal": tn.SamplerWrapper(self.node_s),
            "bkg1": tn.SamplerWrapper(self.node_b),
            "bkg2": tn.SamplerWrapper(self.node_b2),
        }

        batch_size = 20

        faulty_weight_dicts = [
            {
                "signal": 2.,
                "bkg1": 1.,
            },
            {
                "signal": 1.,
                "bkg1": {
                    "bkg2": 1.,
                },
            },
            {
                "signal": 1.,
                "background": 1.,
                "bkg2": 1.,
            },
            {
                "signal": 1.,
                "background": {
                    "bkg1": [0.5, 0.25],
                    "bkg2": [0.15, 0.10],
                },
            },
        ]
        for weight_dict in faulty_weight_dicts:
            with self.subTest(weight_dict=weight_dict):
                with self.assertRaises(ValueError):
                    composite_batched_sampler = BatchedMultiNodeWeightedSampler(
                        node_dict,
                        weights=weight_dict,
                        batch_size=batch_size,
                    )

        allowed_weight_dicts = [
            {
                "signal": 1.,
                "background": {
                    "bkg1": 0.5,
                    "bkg2": 0.5,
                },
            },
        ]
        for weight_dict in allowed_weight_dicts:
            with self.subTest(weight_dict=weight_dict):
                composite_batched_sampler = BatchedMultiNodeWeightedSampler(
                    node_dict,
                    weights=weight_dict,
                    batch_size=batch_size,
                )

                batch_composition = int(batch_size // 2)
                self.assertDictEqual(
                    composite_batched_sampler._batch_composition,
                    {
                        "signal": batch_composition,
                        "background": batch_composition,
                    },
                )
                self.assertListEqual(
                    composite_batched_sampler._weight_samplers,
                    ["background"],
                )
