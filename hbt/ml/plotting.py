# coding: utf-8

from __future__ import annotations

import law
import order as od

from columnflow.util import maybe_import, DotDict


np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
hist = maybe_import("hist")
tf = maybe_import("tensorflow")
shap = maybe_import("shap")


def plot_loss(history, output, classification="categorical") -> None:
    """
    Simple function to create and store a loss plot
    """
    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)

    fig, ax = plt.subplots()
    ax.plot(history["loss"])
    ax.plot(history["val_loss"])
    ax.set(**{
        "ylabel": "Loss",
        "xlabel": "Epoch",
    })
    ax.legend(["train", "validation"], loc="best")
    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False)

    output.child("Loss.pdf", type="f").dump(fig, formatter="mpl")


def plot_accuracy(history, output, classification="categorical") -> None:
    """
    Simple function to create and store an accuracy plot
    """
    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)

    fig, ax = plt.subplots()
    ax.plot(history[f"{classification}_accuracy"])
    ax.plot(history[f"val_{classification}_accuracy"])
    ax.set(**{
        "ylabel": "Accuracy",
        "xlabel": "Epoch",
    })
    ax.legend(["train", "validation"], loc="best")
    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False)

    output.child("Accuracy.pdf", type="f").dump(fig, formatter="mpl")


def plot_confusion(
        model: tf.keras.models.Model,
        inputs: DotDict,
        output: law.FileSystemDirectoryTarget,
        input_type: str,
        process_insts: tuple[od.Process],
) -> None:
    """
    Simple function to create and store a confusion matrix plot
    """
    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Create confusion matrix and normalizes it over predicted (columns)
    confusion = confusion_matrix(
        y_true=np.argmax(inputs['target'], axis=1),
        y_pred=np.argmax(inputs['prediction'], axis=1),
        sample_weight=inputs['weights'],
        normalize="true",
    )

    labels_ext = [proc_inst.label for proc_inst in process_insts] if process_insts else None
    labels = [label.split("HH_{")[1].split("}")[0] for label in labels_ext]
    labels = ["$HH_{" + label for label in labels]
    labels = [label + "}$" for label in labels]

    # Create a plot of the confusion matrix
    fig, ax = plt.subplots(figsize=(15, 10))
    ConfusionMatrixDisplay(confusion, display_labels=labels).plot(ax=ax)

    ax.set_title(f"{input_type} set, rows normalized", fontsize=25, loc="left")
    # plt.imshow(confusion)
    # plt.clim(0, 1)
    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False, loc=2)

    output.child(f"Confusion_{input_type}.pdf", type="f").dump(fig, formatter="mpl")


def plot_roc_ovr(
        inputs: DotDict,
        output: law.FileSystemDirectoryTarget,
        input_type: str,
        process_insts: tuple[od.Process],
) -> None:
    """
    Simple function to create and store some ROC plots;
    mode: OvR (one versus rest)
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    auc_scores = []
    n_classes = len(inputs['target'][0])

    fig, ax = plt.subplots()
    for i in range(n_classes):
        fpr, tpr, thresholds = roc_curve(
            y_true=inputs['target'][:, i],
            y_score=inputs['prediction'][:, i],
            sample_weight=inputs['weights'],
        )

        auc_scores.append(roc_auc_score(
            inputs['target'][:, i], inputs['prediction'][:, i],
            average="macro", multi_class="ovr",
        ))

        # create the plot
        ax.plot(fpr, tpr)

    ax.set_title(f"ROC OvR, {input_type} set")
    ax.set_xlabel("Background selection efficiency (FPR)")
    ax.set_ylabel("Signal selection efficiency (TPR)")

    # legend
    labels = [proc_inst.label for proc_inst in process_insts] if process_insts else range(n_classes)
    ax.legend(
        [f"Signal: {labels[i]} (AUC: {auc_score:.4f})" for i, auc_score in enumerate(auc_scores)],
        loc="best",
    )
    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False, loc=2)

    output.child(f"ROC_ovr_{input_type}.pdf", type="f").dump(fig, formatter="mpl")


def plot_roc_ovr_binary(
        inputs: DotDict,
        output: law.FileSystemDirectoryTarget,
        input_type: str,
        process_insts: tuple[od.Process],
) -> None:
    """
    Simple function to create and store some ROC plots;
    mode: OvR (one versus rest)
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    auc_scores = []

    fig, ax = plt.subplots()
    fpr, tpr, thresholds = roc_curve(
        y_true=inputs['target_binary'],
        y_score=inputs['prediction_binary'],
        sample_weight=inputs['weights'],
    )

    auc_scores.append(roc_auc_score(
        inputs['target_binary'], inputs['prediction_binary'],
        average="macro", multi_class="ovr",
    ))

    # create the plot
    ax.plot(fpr, tpr)

    ax.set_title(f"ROC OvR, {input_type} set")
    ax.set_xlabel("Background selection efficiency (FPR)")
    ax.set_ylabel("Signal selection efficiency (TPR)")

    # legend
    # labels = [proc_inst.label for proc_inst in process_insts] if process_insts else range(n_classes)
    ax.legend(
        [f"(AUC: {auc_scores[0]:.4f})"],
        loc="best",
    )
    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False, loc=2)

    output.child(f"ROC_ovr_{input_type}.pdf", type="f").dump(fig, formatter="mpl")


def plot_output_nodes(
        model: tf.keras.models.Model,
        train: DotDict,
        validation: DotDict,
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
) -> None:
    """
    Function that creates a plot for each ML output node,
    displaying all processes per plot.
    """
    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)

    n_classes = len(train['target'][0])

    colors = ['red', 'blue', 'green', 'orange', 'cyan', 'purple', 'yellow', 'magenta']

    for i in range(n_classes):
        fig, ax = plt.subplots()

        var_title = f"Output node {process_insts[i].label}"

        h = (
            hist.Hist.new
            .StrCat(["train", "validation"], name="type")
            .IntCat([], name="process", growth=True)
            .Reg(20, 0, 1, name=var_title)
            .Weight()
        )

        for input_type, inputs in (("train", train), ("validation", validation)):
            for j in range(n_classes):
                mask = (np.argmax(inputs['target'], axis=1) == j)
                fill_kwargs = {
                    "type": input_type,
                    "process": j,
                    var_title: inputs['prediction'][:, i][mask],
                    "weight": inputs['weights'][mask],
                }
                h.fill(**fill_kwargs)

        plot_kwargs = {
            "ax": ax,
            "label": [proc_inst.label for proc_inst in process_insts],
            "color": colors[:n_classes],
        }

        # dummy legend entries
        plt.hist([], histtype="step", label="Training", color="black")
        plt.hist([], histtype="step", label="Validation (scaled)", linestyle="dotted", color="black")

        # plot training scores
        h[{"type": "train"}].plot1d(**plot_kwargs)

        # legend
        ax.legend(loc="best")

        ax.set(**{
            "ylabel": "Entries",
            "ylim": (0.00001, ax.get_ylim()[1]),
            "xlim": (0, 1),
            # "yscale": 'log',
        })

        # plot validation scores, scaled to train dataset
        scale = h[{"type": "train"}].sum().value / h[{"type": "validation"}].sum().value
        (h[{"type": "validation"}] * scale).plot1d(**plot_kwargs, linestyle="dotted")

        mplhep.cms.label(ax=ax, llabel="Work in progress", data=False, loc=0)
        output.child(f"Node_{process_insts[i].name}.pdf", type="f").dump(fig, formatter="mpl")


def plot_significance(
        model: tf.keras.models.Model,
        train: DotDict,
        validation: DotDict,
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
) -> None:
    plt.style.use(mplhep.style.CMS)

    n_classes = len(train['target'][0])

    store_dict = {}

    for i in range(n_classes):
        fig, ax = plt.subplots()

        for input_type, inputs in (("train", train), ("validation", validation)):
            for j in range(2):
                mask = inputs['target'][:, i] == j
                store_dict[f'node_{process_insts[i].name}_{j}_{input_type}_pred'] = inputs['prediction'][:, i][mask]
                store_dict[f'node_{process_insts[i].name}_{j}_{input_type}_weight'] = inputs['weights'][mask]

        n_bins = 10
        step_size = 1.0 / n_bins
        stop_val = 1.0 + step_size
        bins = np.arange(0.0, stop_val, step_size)
        x_vals = bins[:-1] + step_size / 2
        train_counts_0, train_bins_0 = np.histogram(store_dict[f'node_{process_insts[i].name}_0_train_pred'],
            bins=bins, weights=store_dict[f'node_{process_insts[i].name}_0_train_weight'])
        train_counts_1, train_bins_1 = np.histogram(store_dict[f'node_{process_insts[i].name}_1_train_pred'],
            bins=bins, weights=store_dict[f'node_{process_insts[i].name}_1_train_weight'])
        validation_counts_0, validation_bins_0 = np.histogram(store_dict[f'node_{process_insts[i].name}_0_validation_pred'],
            bins=bins, weights=store_dict[f'node_{process_insts[i].name}_0_validation_weight'])
        validation_counts_1, validation_bins_1 = np.histogram(store_dict[f'node_{process_insts[i].name}_1_validation_pred'],
            bins=bins, weights=store_dict[f'node_{process_insts[i].name}_1_validation_weight'])

        ax.scatter(x_vals, train_counts_1 / np.sqrt(train_counts_0), label="train", color="r")
        ax.scatter(x_vals, validation_counts_1 / np.sqrt(validation_counts_0), label="validation", color="b")
        # title = "$" + process_insts[i].label.split(" ")[2]
        # ax.set_title(f"Significance Node {title}")
        ax.set_ylabel(r"$S/\sqrt{B}$")
        ax.set_xlabel(f"Significance Node {process_insts[i].label}")
        ax.legend(frameon=True)

        mplhep.cms.label(ax=ax, llabel="Work in progress", data=False, loc=0)
        output.child(f"Significance_Node_{process_insts[i].name}.pdf", type="f").dump(fig, formatter="mpl")


def plot_shap_values_simple_nn(
        model: tf.keras.models.Model,
        train: DotDict,
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
        target_dict,
        feature_names,
) -> None:

    feature_dict = {
        "mjj": r"$m_{jj}$",
        "mbjetbjet": r"$m_{bb}$",
        "mHH": r"$m_{HH}$",
        "mtautau": r"$m_{\tau\tau}$",
        "jets_max_d_eta": r"max $\Delta \eta$",
        "jets_d_eta_inv_mass": r"$m_{jj, \Delta \eta}$",
    }

    # names of features and classes
    feature_list = [feature_dict[feature] for feature in feature_names[1]]

    # make sure class names are sorted correctly in correspondence to their target index
    classes = sorted(target_dict.items(), key=lambda x: x[1])
    class_sorted = np.array(classes)[:, 0]
    class_list = ['empty' for i in range(len(process_insts))]
    for proc in process_insts:
        idx = np.where(class_sorted == proc.name)
        class_list[idx[0][0]] = proc.label

    # calculate shap values
    inp = tf.squeeze(train['inputs2'], axis=1).numpy()
    explainer = shap.KernelExplainer(model, inp[:500])
    shap_values = explainer.shap_values(inp[-100:])

    # Feature Ranking
    fig1 = plt.figure()
    shap.summary_plot(shap_values, inp[:100], plot_type="bar",
        feature_names=feature_list, class_names=class_list)
    output.child("Feature_Ranking.pdf", type="f").dump(fig1, formatter="mpl")

    # Violin Plots
    for i, node in enumerate(class_list):
        fig2 = plt.figure()
        shap.summary_plot(shap_values[i], inp[:100], plot_type="violin",
            feature_names=feature_list, class_names=node)
        output.child(f"Violin_{class_sorted[i]}.pdf", type="f").dump(fig2, formatter="mpl")


def plot_shap_values_deep_sets(
        model: tf.keras.models.Model,
        train: DotDict,
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
        target_dict,
        feature_names
) -> None:

    # names of the features
    feature_list = feature_names

    # make sure class names are sorted correctly in correspondence to their target index
    classes = sorted(target_dict.items(), key=lambda x: x[1])
    class_sorted = np.array(classes)[:, 0]
    class_list = ['empty' for i in range(len(process_insts))]
    for proc in process_insts:
        idx = np.where(class_sorted == proc.name)
        class_list[idx[0][0]] = proc.label

    # calculate shap values
    inp = tf.squeeze(train['inputs2'], axis=1).numpy()
    explainer = shap.DeepExplainer(model, inp[:500])
    shap_values = explainer.shap_values(inp[-100:])


def write_info_file(
        output: law.FileSystemDirectoryTarget,
        agg_funcs,
        nodes_deepSets,
        nodes_ff,
        n_output_nodes,
        batch_norm_deepSets,
        batch_norm_ff,
        feature_names,
        process_insts,
        activation_func_deepSets,
        activation_func_ff
) -> None:

    # write info on model for the txt file
    txt_input = f'Processes: {[process_insts[i].name for i in range(len(process_insts))]}\n'
    txt_input += 'Input Handling: Standardization Z-Score \n'
    txt_input += f'Input Features Deep Sets: {feature_names[0]}\n'
    txt_input += f'Input Features FF: {feature_names[1]}\n'
    txt_input += f'Aggregation Functions: {agg_funcs} \n'
    txt_input += 'Deep Sets Architecture:\n'
    txt_input += f'Layers: {len(nodes_deepSets)}, Nodes: {nodes_deepSets}, Activation Function: {activation_func_deepSets}, Batch Norm: {batch_norm_deepSets}\n'
    txt_input += 'FF Architecture:\n'
    txt_input += f'Layers: {len(nodes_ff)}, Nodes: {nodes_ff}, Activation Function: {activation_func_ff}, Batch Norm: {batch_norm_ff}\n'

    output.child('model_specifications.txt', type="d").dump(txt_input, formatter="text")
