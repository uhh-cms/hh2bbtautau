import law.decorator
from columnflow.tasks.union import UniteColumns, UniteColumnsWrapper
from columnflow.util import dev_sandbox, DotDict, maybe_import
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.selection import MergeSelectionStats

from hbt.tasks.base import HBTTask
# from hbt.ml.pytorch_util import ListDataset, MapAndCollate
import law

logger = law.logger.get_logger(__name__)

dd = maybe_import("dask.dataframe")
torch = maybe_import("torch")
torchdata = maybe_import("torchdata")



def run_task(datasets=["hh_ggf_hbb_htt_kl1_kt1_powheg", "tt_sl_powheg"]):
    task = UniteColumnsWrapper(
        **dict((
            ("configs", "run3_2022_postEE_limited"),
            ("version", "pytorch_test"),
            ("datasets", ",".join(datasets)),
        )),
    )

    task.law_run(
        _global=[
            "--workers", 4,
            "--cf.UniteColumns-check-finite-output", "False",
        ],
    )

    sub_targets = {
        d: UniteColumns(
            config = "run3_2022_postEE_limited",
            version = "pytorch_test",
            dataset = d,
        ).target()
        for d in datasets
    }

    from IPython import embed; embed(header="finished running task")
    return task.target()

def main():
    
    output_paths = run_task()
    from IPython import embed; embed(header=f"output_paths: {output_paths}")


class HBTPytorchTask(
    HBTTask,
    UniteColumnsWrapper,
):
    """
    Base task for trigger related studies.
    """

    sandbox = dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_torch.sh")

    reqs = Requirements(
        RemoteWorkflow.reqs,
        UniteColumnsWrapper=UniteColumnsWrapper,
        UniteColumns=UniteColumns,
        MergeSelectionStats=MergeSelectionStats,
    )

    def create_branch_map(self):
        # dummy branch map
        return [0]

    def workflow_requires(self):
        reqs = super().workflow_requires()

        # require the full merge forest
        reqs["unite_columns"] = self.reqs.UniteColumnsWrapper.req(self)
        from IPython import embed; embed(header="workflow_requires")
        return reqs

    def requires(self):
        reqs = DotDict()
        reqs["events"] = DotDict.wrap({
            d: DotDict.wrap({
                config:  self.reqs.UniteColumns.req(self, dataset=d, config=config, _exclude=["datasets", "configs"])
                for config in self.configs
            })
            for d in self.datasets
        })
        # also require selection stats

        reqs["selection_stats"] = DotDict.wrap({
            d: DotDict.wrap({
                config: self.reqs.MergeSelectionStats.req(self, dataset=d, config=config, _exclude=["datasets", "configs"])
                for config in self.configs
            })
            for d in self.datasets
        })

        # from IPython import embed; embed(header="requires")
        return reqs
    
    def output(self):
        return {"model": self.target("dummy.txt")}
    
    def complete(self):
        from law.util import flatten
        # create a flat list of all outputs
        outputs = flatten(self.output())

        if len(outputs) == 0:
            logger.warning("task {!r} has no outputs or no custom complete() method".format(self))
            return True

        return all(t.complete() for t in outputs)

    @law.decorator.log
    @law.decorator.safe_output
    @law.decorator.localize(input=True, output=True)
    def run(self):
        import torchdata.nodes as tn
        from torch.utils.data import RandomSampler, SequentialSampler, default_collate, Dataset
        from torchdata.nodes import MultiNodeWeightedSampler, IterableWrapper

        class ListDataset(Dataset):

            def __init__(self, len: int, prefix: str = "data"):
                self.len = len
                self.prefix = prefix
            
            def __len__(self):
                return self.len
            
            def __getitem__(self, i: int) -> str:
                return f"{self.prefix}_{i}"
            
        class MapAndCollate:
            """A simple transform that takes a batch of indices, maps with dataset, and then applies
            collate.
            TODO: make this a standard utility in torchdata.nodes
            """
            
            def __init__(self, dataset, collate_fn):
                self.dataset = dataset
                self.collate_fn = collate_fn
                
            def __call__(self, batch_of_indices: list[int]):
                batch = [self.dataset[i] for i in batch_of_indices]
                return self.collate_fn(batch)

        print("hello!")

        inputs = self.input()

        signals = ["hh_ggf_hbb_htt_kl1_kt1_powheg"]
        backgrounds = ["tt_sl_powheg"]

        signal_targets = [inputs.events[s].run3_2022_postEE_limited.collection for s in signals]
        backgrounds_targets = [inputs.events[b].run3_2022_postEE_limited.collection for b in backgrounds]

        signal_dfs = dd.read_parquet(
            [
                t.path
                for collections in signal_targets
                for targets in collections.targets.values()
                for t in targets.values() 
            ]
        )
        background_dfs = dd.read_parquet(
            [
                t.path
                for collections in backgrounds_targets
                for targets in collections.targets.values()
                for t in targets.values() 
            ]
        )

        #### test case

        data_s = ListDataset(5, "signal")
        data_b = ListDataset(40, "background")

        foo_s = RandomSampler(data_s)
        foo_b = RandomSampler(data_b)

        foo_s = tn.Batcher(tn.SamplerWrapper(foo_s), batch_size=20, drop_last=False)
        foo_b = tn.Batcher(tn.SamplerWrapper(foo_b), batch_size=20, drop_last=False)
        mapping_s = MapAndCollate(data_s, default_collate)
        mapping_b = MapAndCollate(data_b, default_collate)

        node_s = tn.ParallelMapper(
            foo_s,
            map_fn=mapping_s,
            num_workers=1,
            in_order=True,
            method="process",
        )

        node_b = tn.ParallelMapper(
            foo_b,
            map_fn=mapping_b,
            num_workers=1,
            in_order=True,
            method="process",
        )

        node_dict = {"signal": node_s, "bkg": node_b}

        weight_dict = {"signal": 1., "bkg": 1.}

        node_equal = MultiNodeWeightedSampler(node_dict, weight_dict)



        # good source: https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264
        # https://discuss.pytorch.org/t/proper-way-of-using-weightedrandomsampler/73147
        # https://pytorch.org/data/0.10/stateful_dataloader_tutorial.html
        from IPython import embed
        embed(header="running task")