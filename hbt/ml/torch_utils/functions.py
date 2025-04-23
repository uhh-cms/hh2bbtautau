from __future__ import annotations

from columnflow.util import MockModule, maybe_import

torch = maybe_import("torch")
tqdm = maybe_import("tqdm")
np = maybe_import("numpy")

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

    def get_one_hot(targets, nb_classes):
        # at this point the targets are still ak arrays, so cast to numpy
        targets = targets.to_numpy()
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return np.astype(res.reshape(list(targets.shape) + [nb_classes]), np.int32)

    class WeightedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
        def forward(self, input, target, weight: torch.Tensor | None = None):
            # save original reduction mode
            reduction = self.reduction
            if weight is not None:
                self.reduction = "none"
                loss = super().forward(input, target)
                self.reduction = reduction
                loss = loss * weight
                if self.reduction == "mean":
                    loss = torch.sum(loss) / torch.sum(weight)
                elif self.reduction == "sum":
                    loss = torch.sum(loss)
            else:
                loss = super().forward(input, target)
            return loss
