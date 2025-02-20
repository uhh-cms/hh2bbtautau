import unittest
import os

this_dir = os.path.realpath(os.path.dirname(__file__))

class TorchUtilTests(unittest.TestCase):
    
    def setUp(self):
        from hbt.tasks.studies.torch_util import ListDataset, NodesDataLoader
        from torch.utils.data import default_collate
        import numpy as np
        import json

        # load the results
        with open(os.path.join(this_dir, "torch_util_results_balanced.json"), "r") as f:
            self.balanced_results = json.load(f)

        self.data_s = ListDataset(15, "signal")
        self.data_b = ListDataset(43, "background")
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

    def _compare_values(self, node_dict, weight_dict, results):
        from hbt.tasks.studies.torch_util import BatchedMultiNodeWeightedSampler

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
        from hbt.tasks.studies.torch_util import BatchedMultiNodeWeightedSampler
        import json
        
        # load the results
        with open(os.path.join(this_dir, "torch_util_results_unbalanced.json"), "r") as f:
            results = json.load(f)
        
        node_dict = {"signal": tn.SamplerWrapper(self.node_s), "bkg": tn.SamplerWrapper(self.node_b)}

        self._compare_values(node_dict, {"signal": 2., "bkg": 1.}, results)