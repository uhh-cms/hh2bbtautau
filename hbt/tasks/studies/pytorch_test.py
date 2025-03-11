from __future__ import annotations
import law.decorator
from collections import Collection
from columnflow.tasks.union import UniteColumns, UniteColumnsWrapper
from columnflow.util import dev_sandbox, DotDict, maybe_import
from columnflow.columnar_util import Route
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
np = maybe_import("numpy")
tqdm = maybe_import("tqdm")


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
            CompositeDataLoader, ParquetDataset, FlatParquetDataset,
            NestedDictMapAndCollate,
        )
        from torch.utils.data import default_collate
        import torchdata.nodes as tn
        from hbt.ml.torch_transforms import AkToTensor
        from hbt.ml.torch_models import FeedForwardNet
        
        print("hello!")

        inputs = self.input()
        configs = self.configs
        open_options = {}

        def split_training_validation(
            target_paths,
            ratio=0.7,
            open_options: dict | None = None,
            columns: Collection[str | Route] | None = None,
            targets: Collection[str | int | Route] | str | Route | int | None = None,
            transformations: torch.nn.Module | None = None,
        ):
            meta = ak.metadata_from_parquet(target_paths)
            total_row_groups = meta["num_row_groups"]
            max_training_group = int( total_row_groups*ratio)
            training_row_groups = None
            if max_training_group == 0:
                logger.warning(
                    "Could not split into training and validation data"
                    f" number of row groups for '{target_paths}' is  {total_row_groups}"
                )
            else:
                training_row_groups = np.arange(max_training_group)

            final_options = open_options or dict()

            logger.info(f"Constructing training dataset for {dataset} with row_groups {training_row_groups}")
            training = FlatParquetDataset(
                target_paths,
                open_options=final_options.update({"row_groups": training_row_groups}),
                columns=columns,
                target=targets,
                transform=transformations,
            )

            validation = None
            if training_row_groups is None:
                validation = training
            else:
                validation_row_groups = np.arange(max_training_group, total_row_groups)
                logger.info(f"Constructing validation dataset for {dataset} with row_groups {validation_row_groups}")
                validation = FlatParquetDataset(
                    target_paths,
                    open_options=final_options.update({"row_groups": validation_row_groups}),
                    columns=columns,
                    target=targets,
                    transform=transformations,
                )
            return training, validation

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
        
        model = FeedForwardNet()
        columns = model.inputs
        # construct datamap
        training_data_map = dict()
        validation_data_map = dict()
        for dataset in self.datasets:
            targets = [inputs.events[dataset][c].collection for c in configs]
            target_paths = [
                t.path
                for collections in targets
                for targets in collections.targets.values()
                for t in targets.values()
            ]
            training, validation = split_training_validation(
                target_paths=target_paths, open_options=open_options,
                columns=columns, transformations=AkToTensor(),
                targets=int(1) if dataset.startswith("hh") else int(0)
            )
            training_data_map[dataset] = training
            validation_data_map[dataset] = validation

        
        # extract ttbar sub phase space
        tt_datasets = [d for d in self.datasets if d.startswith("tt_")]
        def extract_probability(dataset: str, keyword: str = "sum_mc_weight_selected"):
            expected_events = list()
            sel_stat = self.input().selection_stats
            for config in self.configs:
                config_inst = self.analysis_inst.get_config(config)
                lumi = config_inst.x.luminosity.nominal
                target = sel_stat[dataset][config].collection[0]["stats"]
                stats = target.load(formatter="json")
                xs = stats.get(keyword, 0)
                expected_events.append(xs * lumi)
            return sum(expected_events)
        weight_dict: dict[str, float | dict[str, float]] = {
            d: 1. for d in self.datasets if not d in tt_datasets
        }
        ttbar_probs = {
            d: extract_probability(d) for d in tt_datasets
        }
        ttbar_prob_sum = sum(ttbar_probs.values())
        weight_dict["ttbar"] = {
            d: val/ttbar_prob_sum for d, val in ttbar_probs.items()
        }

        training_composite_loader = CompositeDataLoader(
            data_map=training_data_map, weight_dict=weight_dict,
            map_and_collate_cls=NestedDictMapAndCollate,
            batch_size=1024,
        )
        validation_composite_loader = CompositeDataLoader(
            data_map=validation_data_map, weight_dict=weight_dict,
            map_and_collate_cls=NestedDictMapAndCollate,
            batch_size=1024,
        )

        def train_loop(dataloader, model, loss_fn, optimizer, update_interval=100):
            # Set the model to training mode - important for batch normalization and dropout layers
            # Unnecessary in this situation but added for best practices
            model.train()
            source_node_names = sorted(dataloader.batcher.source_nodes.keys())
            process_bar: tqdm.std.tqdm = tqdm.tqdm(enumerate(dataloader.data_loader, start=1))
            for ibatch, (X, y) in process_bar:
                # Compute prediction and loss
                pred = model(X)
                target = y["categorical_target"].to(torch.float32)
                loss = loss_fn(pred.squeeze(1), target)
                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if int(ibatch) % int(update_interval) == 0:
                    loss = loss.item()
                    update = f"loss: {loss:>7f} "
                    node_stats = list()
                    for node in source_node_names:
                        n_yielded = dataloader.batcher.source_nodes[node].state_dict()["_num_yielded"]
                        total = len(dataloader.data_map[node])
                        node_stats.append(f"{node}: {n_yielded:>5d} / {total:>5d}")
                    update += "[ {} ]".format(" | ".join(node_stats))
                    process_bar.set_description(update)
        
        def test_loop(dataloader, model, loss_fn):
            # Set the model to evaluation mode - important for batch normalization and dropout layers
            # Unnecessary in this situation but added for best practices
            model.eval()
            size = len(dataloader)
            # num_batches = dataloader.num_batches
            test_loss, correct = 0, 0

            # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
            # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
            with torch.no_grad():
                num_batches = 0
                process_bar = tqdm.tqdm(dataloader.data_loader, desc="Validation")
                for X, y in process_bar:
                    pred = model(X)
                    target = y["categorical_target"].to(torch.float32)
                    test_loss += loss_fn(pred.squeeze(1), target).item()
                    correct += (pred.argmax(1) == target).type(torch.float).sum().item()
                    num_batches += 1

            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
        logger.info("Constructing loss and optimizer")
        loss_fn = torch.nn.CrossEntropyLoss()
        from torch.optim import Adam
        optimizer = Adam(model.parameters(), lr=1e-3)

        from IPython import embed
        embed(header="initialized model")
        # good source: https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264
        # https://discuss.pytorch.org/t/proper-way-of-using-weightedrandomsampler/73147
        # https://pytorch.org/data/0.10/stateful_dataloader_tutorial.html
        