from __future__ import annotations

__all__ = [
    "network_predictions",
    "confusion_matrix",
]

from columnflow.util import MockModule, maybe_import

torch = maybe_import("torch")
tqdm = maybe_import("tqdm")
np = maybe_import("numpy")
ak = maybe_import("awkward")
sklearn = maybe_import("sklearn")
plt = maybe_import("matplotlib.pyplot")

if not isinstance(torch, MockModule):

    def plot_input_features(data_map, columns):
        from columnflow.columnar_util import Route, EMPTY_INT, EMPTY_FLOAT
        all_data = ak.concatenate([x.data for x in data_map])
        columns = list(map(Route, columns))
        # layout plots
        num_features = len(columns)
        num_cols = 4
        num_row = int(np.ceil(num_features / num_cols))
        fig_size = (5 * num_cols, 4 * num_row) # wide, tall

        fig, axes = plt.subplots(nrows=num_row, ncols=num_cols, figsize=fig_size)
        for ax, _route in zip(axes.flatten(), columns):
            data = _route.apply(all_data).to_numpy().astype(np.float32)
            # mask NaNs
            empty_mask = data == EMPTY_FLOAT

            ax.set_xlabel(_route.column)
            ax.set_ylabel("frequency")
            ax.set_yscale("log")

            # get lowest value without empty values and add offset
            _bin = 20
            bins = np.linspace(
                np.min(data[~empty_mask]),
                data.max(),
                _bin,
            )
            # set offset to 3 bins to display underflow, clip data to lower edge and preserve bin width
            lower_edge = bins[0] - 3 * (bins[1] - bins[0])
            bins = np.linspace(lower_edge, bins[-1], _bin + 3)

            # plot without empty values, set empty values to underflow bin
            _ = ax.hist(np.clip(data, a_min=lower_edge, a_max=None), bins=bins)
        return fig, axes



    def _network_predictions(y_true, y_pred, target_map, **kwargs):
        # swap axes to get (num_samples, cls) shape
        fig, ax = plt.subplots()
        ax.set_xlabel(kwargs.pop("xlabel", "score"))
        ax.set_ylabel(kwargs.pop("ylabel", "frequency"))
        fig.suptitle(kwargs.pop("title", None))

        # get events that are predicted correctly for each class
        y_pred_cls, labels = [], []
        for _cls, idx in target_map.items():
            # get events of cls
            correct_cls_mask = y_true[:, idx] == 1
            # filter predictions for cls
            y_pred_cls.append(y_pred[correct_cls_mask][:, 0])
            labels.append(f"{_cls}")

        _ = ax.hist(
            y_pred_cls,
            bins=kwargs.get("bins", 20),
            histtype=kwargs.get("histtype", "step"),
            alpha=kwargs.get("alpha", 0.7),
            label=labels,
            **kwargs,
        )

        fig.legend(title="predicted class")
        return fig, ax

    def network_predictions(y_true, y_pred, target_map, **kwargs):
        # create a figure with subplots for each node, where the score is shown
        # nodes are defined over target_map order

        fig, axes = plt.subplots(1,len(target_map), figsize=(8 * len(target_map), 8))
        fig.suptitle(kwargs.pop("title", None))

        # get events that are predicted correctly for each class
        for node, node_idx in target_map.items():
            for data_cls, data_idx in target_map.items():
                axes[node_idx].set_xlabel(f"{node} node")
                axes[node_idx].set_ylabel("frequency")
                axes[node_idx].set_yscale("log")

                # get events of specific cls (e.g. hh)
                correct_cls_mask = y_true[:, data_idx] == 1
                # get predictions for cls
                filtered_predictions = y_pred[correct_cls_mask][:, node_idx]


                _ = axes[node_idx].hist(
                    filtered_predictions,
                    bins=kwargs.get("bins", 20),
                    histtype=kwargs.get("histtype", "step"),
                    alpha=kwargs.get("alpha", 0.7),
                    label=data_cls,
                    **kwargs,
            )
            axes[node_idx].set_ylim(top=len(y_pred))



        lines_labels = [fig.axes[0].get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels)
        return fig, axes


    def confusion_matrix(y_true, y_pred, target_map, sample_weight=None, cmap="Blues", **kwargs):
        cm = sklearn.metrics.confusion_matrix(
            y_true,
            y_pred,
            labels=list(target_map.values()),
            sample_weight=sample_weight,
            normalize="true",  # normalize to get probabilities
        )

        disp = sklearn.metrics.ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=list(target_map.keys()),
        )
        disp.plot(cmap=cmap)
        disp.figure_.suptitle(kwargs.pop("title", None))

        return disp.figure_, disp.ax_, confusion_matrix

    def plot_batch(self, input, target, loss, iteration, target_map=None, labels=None):
        input_per_feature = input.to("cpu").transpose(0, 1).detach().numpy()
        input, target = input.to("cpu").detach().numpy(), target.to("cpu").detach().numpy()

        fig, ax = plt.subplots(1, len(self.categorical_inputs), figsize=(8 * len(self.categorical_inputs), 8))
        fig.tight_layout()

        for ind, cat in enumerate(self.categorical_inputs):
            signal_target = target[:, self.categorical_target_map["hh"]]

            background_mask, signal_mask = signal_target.flatten() == 0, signal_target.flatten() == 1
            # background_prediction = detach_pred[zero_s_mask]
            # signal_prediction = detach_pred[zero_s_mask]
            _input = input_per_feature[ind]
            background_input = _input[background_mask]
            signal_input = _input[signal_mask]
            cax = ax[ind]

            cax.hist(
                [background_input, signal_input],
                bins=10,
                histtype="barstacked",
                alpha=0.5,
                label=["tt & dy", "hh"],
                density=True,
            )
            cax.set_xlabel(cat)
            cax.annotate(f"loss: {loss:.2f}", (0.8, 0.60), xycoords="figure fraction")
            cax.annotate(f"iteration {iteration}", (0.75, 0.55))
            cax.legend()
        fig.savefig(f"1d_cats_{iteration}.png")

    def plot_2D(x, y, bins=10, **kwargs):
        fig, ax = plt.subplots()
        hist = ax.hist2d(
            x,
            y,
            bins=bins,
            cmap=kwargs.get("cmap"),
        )
        ax.set_xlabel(kwargs.get("xlabel"))
        ax.set_xlabel(kwargs.get("ylabel"))
        fig.colorbar(hist[3], ax=ax)
        ax.set_title(kwargs.get("title"))
        if kwargs.get("savepath"):
            fig.savefig(kwargs.get("savepath"))
        return fig, ax

    def plot_1D(x, annotations=None, **kwargs):
        fig, ax = plt.subplots()
        _ = ax.hist(
            x,
            bins=kwargs.get("bins"),
            histtype="stepfilled",
            alpha=0.7,
            color=kwargs.get("color"),
            label=kwargs.get("label"),
            density=kwargs.get("density"),
        )
        ax.set_xlabel(kwargs.get("xlabel"))
        ax.set_xlabel(kwargs.get("ylabel"))
        ax.set_title(kwargs.get("title"))
        if kwargs.get("savepath"):
            fig.savefig(kwargs.get("savepath"))
        ax.legend()

        if annotations:
            loss = annotations.get("loss")
            target = annotations.get("target")
            pred = annotations.get("pred")
            num_0s = np.sum(target == 0)
            num_1s = np.sum(target == 1)

            ax.annotate(f"num: 0s: {num_0s:.2f}", (0.5, 0.70), xycoords="axes fraction")
            ax.annotate(f"num: 1s: {num_1s:.2f}", (0.5, 0.65), xycoords="axes fraction")
            ax.annotate(f"loss: {loss:.2f}", (0.5, 0.60), xycoords="axes fraction")
            iteration = annotations.get("iteration")
            ax.annotate(f"iteration {iteration}", (0.5, 0.55))

            tp = np.sum((target == 1) & (pred > 0.5))
            tn = np.sum((target == 0) & (pred < 0.5))
            fp = np.sum((target == 0) & (pred > 0.5))
            fn = np.sum((target == 1) & (pred < 0.5))
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            sensitivity = tp / (tp + fn)

            ax.annotate(f"accuracy: {accuracy:.2f}", (0.5, 0.50), xycoords="axes fraction")
            ax.annotate(f"sensitivity: {sensitivity:.2f}", (0.5, 0.45), xycoords="axes fraction")

        return fig, ax

    def control_plot_1d(train_loader, dataset_handler):
        import matplotlib.pyplot as plt
        d = {}
        for dataset, file_handler in training_loader.data_map.items():
            d[dataset] = ak.concatenate(list(map(lambda x: x.data, file_handler)))

        for cat in dataset_handler.categorical_featuress:
            plt.clf()
            data = []
            labels = []
            for dataset, arrays in d.items():
                data.append(Route(cat).apply(arrays))
                labels.append(dataset)
            plt.hist(data, histtype="barstacked", alpha=0.5, label=labels)
            plt.xlabel(cat)
            plt.legend()
            plt.savefig(f"{cat}_all.png")
