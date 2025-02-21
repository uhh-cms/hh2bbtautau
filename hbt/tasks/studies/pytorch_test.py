from __future__ import annotations
import law.decorator
from columnflow.tasks.union import UniteColumns, UniteColumnsWrapper
from columnflow.util import dev_sandbox, DotDict, maybe_import
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.selection import MergeSelectionStats
from columnflow.types import Callable

from hbt.tasks.base import HBTTask
# from hbt.ml.pytorch_util import ListDataset, MapAndCollate
import law

logger = law.logger.get_logger(__name__)

dd = maybe_import("dask.dataframe")
torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
dak = maybe_import("dask_awkward")
ak = maybe_import("awkward")




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
        from hbt.tasks.studies.torch_util import (
            NodesDataLoader, ParquetDataset, BatchedMultiNodeWeightedSampler
        )
        from torch.utils.data import default_collate
        import torchdata.nodes as tn
        
        print("hello!")

        inputs = self.input()

        signals = ["hh_ggf_hbb_htt_kl1_kt1_powheg"]
        backgrounds = ["tt_sl_powheg"]

        configs = self.configs
        columns = ["Jet.*", "Muon.*", "Electron.pt"]
        open_options = {}

        signal_targets = [inputs.events[s][configs[0]].collection for s in signals]
        signal_target_paths = [
            t.path
            for collections in signal_targets
            for targets in collections.targets.values()
            for t in targets.values()
        ]
        backgrounds_targets = [inputs.events[b][configs[0]].collection for b in backgrounds]
        background_target_paths = [
            t.path
            for collections in backgrounds_targets
            for targets in collections.targets.values()
            for t in targets.values()
        ]

        ### read via dask dataframe ######################################################
        # signal_dfs = dd.read_parquet(
        #     [
        #         t.path
        #         for collections in signal_targets
        #         for targets in collections.targets.values()
        #         for t in targets.values() 
        #     ]
        # )
        # background_dfs = dd.read_parquet(
        #     [
        #         t.path
        #         for collections in backgrounds_targets
        #         for targets in collections.targets.values()
        #         for t in targets.values() 
        #     ]
        # )

        ### read via dask awkward ###################################################
        # signal_daks = dak.from_parquet(
        #     signal_target_paths,
        #     split_row_groups=True,
        # )

        # pro:
        # - can read multiple parquet files w/o loading to memory
        # - can read only the columns we need
        # - option to split between row groups
        # con:
        # - when accessing single elements, there seems to be a lot of overhead/
        #   leaked memory
        # - each compute step takes time


        #### read via awkward array #####################################################
        # signal_daks = ak.from_parquet(
        #     signal_target_paths,
        # )

        # pro:
        # - can read multiple parquet files
        # - can read only the columns we need
        # - fast
        # - can also read individual partitions (not implemented now)
        # con:
        # - eager loading of all data (problem?)

        #### test case

        data_s = ParquetDataset(
            signal_target_paths,
            open_options=open_options,
            columns=columns,
        )
        data_b = ParquetDataset(
            background_target_paths,
            open_options=open_options,
            columns=columns,
        )

        node_s = NodesDataLoader(
            data_s,
            shuffle=True,
            num_workers=1,
            collate_fn=lambda x: x,
            pin_memory=False,
        )
        node_b = NodesDataLoader(
            data_b,
            shuffle=True,
            num_workers=1,
            collate_fn=lambda x: x,
            pin_memory=False,
        )

        # down-sides of this approach:
        # - number of batches can vary
        # - batches are not balanced - only average over all batches is balanced

        node_dict = {
            "signal": tn.SamplerWrapper(node_s),
            "background": tn.SamplerWrapper(node_b),
        }
        weight_dict = {
            "signal": 1.,
            "background": 1.,
        }

        composite_batched_sampler = BatchedMultiNodeWeightedSampler(
            node_dict,
            weights=weight_dict,
            batch_size=256,
        )

        from IPython import embed
        embed(header="running task")

        composite_batched_sampler.reset()
        
        # get first element
        foo = composite_batched_sampler.next()
        # good source: https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264
        # https://discuss.pytorch.org/t/proper-way-of-using-weightedrandomsampler/73147
        # https://pytorch.org/data/0.10/stateful_dataloader_tutorial.html
        