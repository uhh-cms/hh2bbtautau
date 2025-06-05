from __future__ import annotations

__all__ = [
    "get_one_hot",
    "preprocess_multiclass_outputs",
    "plot_confusion_matrix",
    "generate_weighted_loss",
    "WeightedCrossEntropyLoss",
    "WeightedCrossEntropySlice",
]

from columnflow.util import MockModule, maybe_import

torch = maybe_import("torch")
tqdm = maybe_import("tqdm")
np = maybe_import("numpy")
ak = maybe_import("awkward")
sklearn = maybe_import("sklearn")

get_one_hot = MockModule("get_one_hot")
preprocess_multiclass_outputs = MockModule("preprocess_multiclass_outputs")
plot_confusion_matrix = MockModule("plot_confusion_matrix")
generate_weighted_loss = MockModule("generate_weighted_loss")
WeightedCrossEntropyLoss = MockModule("WeightedCrossEntropyLoss")
WeightedCrossEntropySlice = MockModule("WeightedCrossEntropySlice")

if not isinstance(torch, MockModule):

    def train_loop(dataloader, model, loss_fn, optimizer, update_interval=100):
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.train()
        source_node_names = sorted(dataloader.batcher.source_nodes.keys())
        process_bar: tqdm.std.tqdm = tqdm.tqdm(enumerate(dataloader.data_loader, start=1))
        for ibatch, (X, y) in process_bar:
            # Compute prediction and loss
            optimizer.zero_grad()
            pred = model(X)
            target = y["categorical_target"].to(torch.float32).reshape(-1, 1)
            # from IPython import embed
            # embed(header=f"training loop in batch {ibatch}")
            loss = loss_fn(pred, target)
            # Backpropagation
            loss.backward()
            optimizer.step()

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
                target = y["categorical_target"].to(torch.float32).reshape(-1, 1)
                test_loss += loss_fn(pred, target).item()
                correct += (pred == target).type(torch.float).sum().item()
                num_batches += 1

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def ignite_train_step(engine, batch, model, loss_fn, optimizer):
        # Set the model to training mode - important for batch normalization and dropout layers
        model.train()
        # Compute prediction and loss
        X, y = batch[0], batch[1]
        optimizer.zero_grad()
        pred = model(X)
        target = y["categorical_target"].to(torch.float32).reshape(-1, 1)
        # from IPython import embed
        # embed(header=f"training loop in batch {ibatch}")
        loss = loss_fn(pred, target)
        # Backpropagation
        loss.backward()
        optimizer.step()

        return loss.item()

    def ignite_validation_step(engine, batch, model):
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        model.eval()

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            X, y = batch[0], batch[1]
            pred = model(X)
            target = y["categorical_target"].to(torch.float32).reshape(-1, 1)
            return pred, target

    def get_one_hot(targets, nb_classes):  # noqa: F811
        # at this point the targets are still ak arrays, so cast to numpy
        if isinstance(targets, ak.Array):
            targets = targets.to_numpy()
        elif isinstance(targets, torch.Tensor):
            targets = targets.numpy()

        indices = np.array(targets, dtype=np.int32).reshape(-1)
        res = np.eye(nb_classes)[indices]
        return np.astype(res.reshape(list(targets.shape) + [nb_classes]), np.int32)

    def preprocess_multiclass_outputs(outputs, **additional_kwargs):  # noqa: F811
        kwargs = dict()
        if len(outputs) == 2:
            y_pred, y_true = outputs
        elif len(outputs) == 3:
            y_pred, y_true, kwargs = outputs

        # fix mismatch in weight naming
        # if "weight" in kwargs.keys():
        #     kwargs["sample_weight"] = kwargs.pop("weight")

        kwargs.update(additional_kwargs)

        # first get softmax from predictions
        y_pred = torch.nn.functional.softmax(y_pred)

        return y_pred, y_true, kwargs

    def preprocess_multiclass_outputs_and_targets(outputs, **additional_kwargs):  # noqa: F811
        kwargs = dict()
        if len(outputs) == 2:
            y_pred, y_true = outputs
        elif len(outputs) == 3:
            y_pred, y_true, kwargs = outputs

        # fix mismatch in weight naming
        # if "weight" in kwargs.keys():
        #     kwargs["sample_weight"] = kwargs.pop("weight")

        kwargs.update(additional_kwargs)

        # first get softmax from predictions
        y_pred = torch.nn.functional.softmax(y_pred)
        if y_true.dim() > 1:
            y_true = torch.argmax(y_true, dim=-1)

        return y_pred, y_true, kwargs

    def plot_confusion_matrix(confusion_matrix, labels=None, sample_weight=None, cmap="Blues", savepath=None):  # noqa: F811

        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
        disp.plot(cmap=cmap)

        if savepath:
            disp.figure_.savefig(savepath)
        return disp.figure_, disp.ax_, confusion_matrix

    def generate_weighted_loss(  # noqa: F811
        loss_fn: torch.nn.Module,
    ):

        class WeightedLoss(loss_fn):
            def forward(self, input, target, weight: torch.Tensor | None = None):
                # save original reduction mode
                reduction = self.reduction

                if weight is not None:
                    self.reduction = "none"
                    loss = super().forward(input, target)
                    self.reduction = reduction

                    # dot product is only defined for flat tensors, so flatten
                    loss = torch.flatten(loss)
                    weight = torch.flatten(weight)
                    loss = torch.dot(loss, weight)
                    if self.reduction == "mean":
                        loss = loss / torch.sum(weight)
                else:
                    loss = super().forward(input, target)
                return loss

        WeightedLoss.__name__ = f"Weighted{loss_fn.__name__}"
        return WeightedLoss

    WeightedCrossEntropyLoss = generate_weighted_loss(torch.nn.CrossEntropyLoss)

    class WeightedCrossEntropySlice(WeightedCrossEntropyLoss):  # noqa: F811
        def __init__(self, cls_index: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cls_index = cls_index

        def forward(self, input, target, weight: torch.Tensor | None = None, **kwargs):
            if weight is None:
                shape = input.shape[:-1] if input.dim() > 1 else input.shape
                weight = torch.ones(shape)
            # save original reduction mode
            reduction = self.reduction
            self.reduction = "none"
            loss = super().forward(input, target)
            self.reduction = reduction
            # select the loss items that belong to the current slice index
            # first, see where the maximum entry in the target vector can be found
            max_idx = torch.max(target, dim=-1).indices

            # create mask that corresponds to current class index and reduce
            # tensors accordingly
            cls_mask = max_idx == self.cls_index
            loss = loss[cls_mask]
            weight = weight[cls_mask]

            # continue with calculation
            loss = torch.dot(loss, weight)
            if self.reduction == "mean" and not loss == 0 and not torch.sum(weight) == 0:
                loss = loss / torch.sum(weight)
            return loss
