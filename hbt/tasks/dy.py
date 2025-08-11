# coding: utf-8

"""
Tasks to study and create weights for DY events.
"""

from __future__ import annotations

import re
import gzip

import law

from columnflow.tasks.framework.base import ConfigTask
from columnflow.tasks.framework.histograms import HistogramsUserSingleShiftBase
from columnflow.util import maybe_import, dev_sandbox

from hbt.tasks.base import HBTTask

hist = maybe_import("hist")


class ComuteDYWeights(HBTTask, HistogramsUserSingleShiftBase):
    """
    Example command:

        > law run hbt.ComuteDYWeights \
            --config 22pre_v14 \
            --processes sm_bkg \
            --version prod8_dy \
            --hist-producer no_dy_weight \
            --categories mumu__dyc__os \
            --variables njets-dilep_pt or njets
    """

    single_config = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # only one category is allowed right now
        if len(self.categories) != 1:
            raise ValueError(f"{self.task_family} requires exactly one category, got {self.categories}")
        # ensure that the category matches a specific pattern: starting with "ee"/"mumu" and ending in "os"
        if not re.match(r"^(mumu)__.*__os.*$", self.categories[0]):
            raise ValueError(f"category must start with '{{mumu}}__' and contain '__os', got {self.categories[0]}")
        self.category_inst = self.config_inst.get_category(self.categories[0])

        # only one variable is allowed
        if len(self.variables) != 1:
            raise ValueError(f"{self.task_family} requires exactly one variable, got {self.variables}")
        self.variable = self.variables[0]
        # for now, variable must be "njets-dilep_pt"
        if self.variable not in ["njets-dilep_pt", "njets"]:
            raise ValueError(f"variable must be either 'njets-dilep_pt' or 'njets', got {self.variable}")

    def output(self):
        return self.target(f"dy_weight_data_{self.categories[0]}_{self.variables[0]}.pkl")

    def run(self):
        # prepare categories to sum over
        leaf_category_insts = self.category_inst.get_leaf_categories() or [self.category_inst]

        # load histograms from all input datasets and add them
        h = None
        for dataset_name, inp in self.input().items():
            h_ = inp.collection[0]["hists"][self.variable].load(formatter="pickle")

            # select leaf categories
            h_ = h_[{"category": [hist.loc(cat.name) for cat in leaf_category_insts if cat.name in h_.axes["category"]]}]

            # use the nominal shift only
            h_ = h_[{"shift": hist.loc("nominal")}]

            # add it
            h = h_ if h is None else (h + h_)

        # compute the dy weight data
        dy_weight_data = compute_weight_data(self, h)  # use --variables njets-dilep_pt
        # dy_weight_data = compute_njet_norm_data(self, h)  # use --variables njets

        # store them
        self.output().dump(dy_weight_data, formatter="pickle")


class ExportDYWeights(HBTTask, ConfigTask):
    """
    Example command:

        > law run hbt.ExportDYWeights \
            --configs 22pre_v14,22post_v14,... \
            --version prod12_nody
    """

    sandbox = dev_sandbox("bash::$CF_BASE/sandboxes/venv_columnar.sh")

    single_config = False

    categories = ("mumu__dyc__os",)
    # variable = "njets-dilep_pt"
    variable = "njets"

    def requires(self):
        return {
            config: ComuteDYWeights.req(
                self,
                config=config,
                processes=("sm_data",),  # supports groups
                # hist_producer="no_dy_weight",
                hist_producer="default",
                categories=self.categories,
                variables=(self.variable,),
            )
            for config in self.configs
        }

    def output(self):
        return self.target(f"hbt_corrections_{self.variable}.json.gz")

    def run(self):
        import correctionlib.schemav2 as cs
        from hbt.studies.dy_weights.create_clib_file import create_dy_weight_correction

        # load all weight data per config and merge them into a single dictionary
        dy_weight_data = law.util.merge_dicts(*(
            inp.load(formatter="pickle") for inp in self.input().values()
        ))

        print(dy_weight_data)

        # create and save the correction set
        cset = cs.CorrectionSet(
            schema_version=2,
            description="Corrections derived for the hh2bbtautau analysis.",
            corrections=[create_dy_weight_correction(dy_weight_data)],
        )
        with self.output().localize("w") as outp:
            outp.path += ".json.gz"
            with gzip.open(outp.abspath, "wt") as f:
                f.write(cset.model_dump_json(exclude_unset=True))

            # validate the content
            law.util.interruptable_popen(f"correction summary {outp.abspath}", shell=True)


def get_ratio_values(h: hist.Hist) -> tuple[hist.Hist, hist.Hist, hist.Hist]:

    import numpy as np

    # get and sum process histograms
    dy_names = [name for name in h.axes["process"] if name.startswith("dy")]
    data_names = [name for name in h.axes["process"] if name.startswith("data")]
    mc_names = [name for name in h.axes["process"] if not name.startswith(("dy", "data"))]
    dy_h = h[{"process": list(map(hist.loc, dy_names))}][{"process": sum}]
    data_h = h[{"process": list(map(hist.loc, data_names))}][{"process": sum}]
    mc_h = h[{"process": list(map(hist.loc, mc_names))}][{"process": sum}]

    # get bin centers
    bin_centers = dy_h.axes[-1].centers

    # get histogram values and errors
    dy_values = dy_h.view().value
    data_values = data_h.view().value
    mc_values = mc_h.view().value
    dy_err = dy_h.view().variance**0.5
    data_err = data_h.view().variance**0.5
    mc_err = mc_h.view().variance**0.5

    # calculate (data-mc)/dy ratio and its error
    ratio_values = (data_values - mc_values) / dy_values
    ratio_err = (1 / dy_values) * np.sqrt(data_err**2 + mc_err**2 + (ratio_values * dy_err)**2)

    # fill nans/infs and negative errors with 0
    ratio_values = np.nan_to_num(ratio_values, nan=0.0)
    ratio_values = np.where(np.isinf(ratio_values), 0.0, ratio_values)
    ratio_values = np.where(ratio_values < 0, 0.0, ratio_values)
    ratio_err = np.nan_to_num(ratio_err, nan=0.0)
    ratio_err = np.where(np.isinf(ratio_err), 0.0, ratio_err)
    ratio_err = np.where(ratio_err < 0, 0.0, ratio_err)

    return (ratio_values, ratio_err, bin_centers)


def compute_weight_data(task: ComuteDYWeights, h: hist.Hist) -> dict:
    """
    Compute the DY weight data from the given histogram *h* that supposed to contain the following axis:

        - process (str)
        - njets (int)
        - dilep_pt (float)

    The returned dictionary follows a nested structure:

        year -> syst -> (min_njet, max_njet) -> [(lower_bound, upper_bound, formula), ...]

        - *year* is one of 2022, 2022EE, 2023, 2023BPix
        - *syst* is one of nominal, up, down (maybe up1, down1, up2, down2, ... in case of multiple sources)
        - *(min_njet, max_njet)* is a tuple of integers defining the right-exclusive range of njets
        - the inner-most list contains 3-tuples with lower and upper bounds of a formula
    """
    # prepare constants
    inf = float("inf")
    import numpy as np
    from scipy import optimize, special
    from matplotlib import pyplot as plt
    from scipy.interpolate import UnivariateSpline
    era = f"{task.config_inst.campaign.x.year}{task.config_inst.campaign.x.postfix}"

    # use scipy erf function definition
    def window(x, r, s):
        """
        x: dependent variable (i.g., dilep_pt)
        r: regime boundary between two fit functions
        s: sign of erf function (+1 to active second fit function, -1 to active first fit function)
        """

        sci_erf = 0.5 * (special.erf(s * 0.08 * (x - r)) + 1)

        return sci_erf

    def fit_function(x, c, n, mu, sigma, a, b, r):

        """
        A fit function.
        x: dependent variable (i.g., dilep_pt)
        c: Gaussian offset
        n: Gaussian normalization
        mu and sigma: Gaussian parameters
        a, b: polinomial parameters
        r: regime boundary between Guassian and linear fits
        """

        gauss = c + (n * (1 / sigma) * np.exp(-0.5 * ((x - mu) / sigma) ** 2))
        pol = a + b * x
        # pol2 = a + b * x + d * (x ** 2)

        return window(x, r, -1) * gauss + window(x, r, 1) * pol

    # build fit function string
    def get_fit_str(njet: int, h: hist.Hist) -> dict:

        ratio_values, ratio_err, bin_centers = get_ratio_values(h)

        # define starting values for c, n, mu, sigma, a, b, r with respective bounds
        starting_values = [1, 1, 10, 3, 1, 0, 50]
        lower_bounds = [0.6, 0, 0, 0, 0, -2, 20]
        upper_bounds = [1.2, 10, 50, 20, 2, 3, 100]

        # perform the fit and get post-fit parameters
        popt, pcov = optimize.curve_fit(
            fit_function,
            bin_centers,
            ratio_values,
            p0=starting_values, method="trf",
            sigma=np.maximum(ratio_err, 1e-5),
            absolute_sigma=True,
            bounds=(lower_bounds, upper_bounds),
        )

        c, n, mu, sigma, a, b, r = popt

        # define string expression of the fit funtion
        for var_name in ['c', 'n', 'mu', 'sigma', 'a', 'b', 'r']:
            locals()[var_name] = f"{locals()[var_name]:.9f}"
        gauss = f"(({c})+(({n})*(1/{sigma})*exp(-0.5*((x-{mu})/{sigma})^2)))"
        pol = f"(({a})+({b})*x)"
        # pol2 = f"(({a})+({b})*x+({d})*(x^2))"
        fit_string = f"(0.5*(erf(-0.08*(x-{r}))+1))*{gauss}+(0.5*(erf(0.08*(x-{r}))+1))*{pol}"

        # -----------------------------------------------------
        # generate toy parameteres for uncertainty estimation
        # -----------------------------------------------------

        pcov_norm = pcov / np.outer(popt, popt)
        eigvals, eigvecs = np.linalg.eigh(pcov_norm)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        total = eigvals.sum()
        explained = np.cumsum(eigvals) / total
        threshold = 0.95
        k = np.searchsorted(explained, threshold) + 1

        # Vk = eigvecs[:, :k]  # (8,k) matrix of eigenvectors
        # from IPython import embed; embed(header="debugger")

        #  plot the fit function + toys
        s = np.linspace(0, 200, 1000)
        y = [fit_function(v, *popt) for v in s]

        for i in range(k):
            y_up = fit_function(s, *(popt * (1 + eigvecs[:, i] * np.sqrt(eigvals[i]))))
            y_down = fit_function(s, *(popt * (1 - eigvecs[:, i] * np.sqrt(eigvals[i]))))
            fig, ax = plt.subplots()
            ax.plot(s, y, color="black", label="Fit", lw=0.8, linestyle="--")
            ax.fill_between(s, y_up, y, color='red', alpha=0.5, label='95% (sys) up')
            ax.fill_between(s, y_down, y, color='blue', alpha=0.5, label='95% (sys) down')
            ax.errorbar(
                bin_centers,
                ratio_values,
                yerr=ratio_err,
                fmt=".",
                color="black",
                linestyle="none",
                ecolor="black",
                elinewidth=0.5,
            )
            ax.legend(loc="upper right")
            ax.set_xlabel(r"$p_{T,ll}\;[\mathrm{GeV}]$")
            ax.set_ylabel("Ratio (data-non DY MC)/DY")
            ax.grid(True)

            if njet != 4:
                if njet == 2:
                    ax.set_ylim(0.75, 1.05)
                else:
                    ax.set_ylim(0.8, 1.2)
                title = f"2022preEE, njets = {njet}, k={i+1}(/{k})"
                file_name = f"plot_dilep_eq{njet}j_gauss_pol2_0.08_k{i+1}.pdf"
            else:
                ax.set_ylim(0.8, 1.5)
                title = f"2022preEE, njets >= {njet}, k={i+1}(/{k})"
                file_name = f"plot_dilep_ge{njet}j_gauss_pol2_0.08_k{i+1}.pdf"

            ax.set_title(title)
            fig.savefig(file_name)

        return fit_string

    # initialize fit dictionary
    fit_dict = {
        era: {
            "nominal": {},
            # "up": {},
            # "down": {},
        },
    }

    # do the fit per njet category
    leaf_cats = h.axes["category"]

    for cat in leaf_cats:
        # Extract njet number from category name (e.g., "eq2j" -> 2)
        match = re.search(r'(\d+)j', cat)
        if match:
            njet = int(match.group(1))
        else:
            raise ValueError("No njets digit found in category name!")

        # slice the histogram for the selected njets bin
        if njet != 4:
            h_ = h[cat, :, njet, ...]
            fit_str = get_fit_str(njet, h_)
            fit_dict[era]["nominal"][(njet, njet + 1)] = [(-inf, inf, fit_str)]
        else:
            h_ = h[cat, ...][{"njets": sum}]
            fit_str = get_fit_str(njet, h_)
            fit_dict[era]["nominal"][(njet, 50)] = [(-inf, inf, fit_str)]

    print("----------")
    print(fit_dict)

    return fit_dict


def compute_njet_norm_data(task: ComuteDYWeights, h: hist.Hist) -> dict:

    # prepare constants
    inf = float("inf")
    import numpy as np
    era = f"{task.config_inst.campaign.x.year}{task.config_inst.campaign.x.postfix}"

    fit_dict = {
        era: {
            "nominal": {},
            "up": {},
            "down": {},
        },
    }

    # do the fit per njet (or nbjet) category
    leaf_cats = h.axes["category"]

    # hist with all leaf categories
    for cat in leaf_cats:
        print("############################")
        print(f"cat: {cat}")
        # Extract njet number from category name
        match = re.search(r'eq(\d+)j', cat)
        if match:
            njet = int(match.group(1))
        elif re.search(r'ge6j', cat):
            njet = 6
        else:
            continue

        # slice the histogram for the selected njets bin
        if njet < 4:
            fit_dict[era]["nominal"][(njet, njet + 1)] = [(-inf, inf, "x*0+1.0")]
            fit_dict[era]["up"][(njet, njet + 1)] = [(-inf, inf, "x*0+1.0")]
            fit_dict[era]["down"][(njet, njet + 1)] = [(-inf, inf, "x*0+1.0")]

        elif njet in [4, 5]:
            h_ = h[cat, ...]
            ratio_values, ratio_err, bin_centers = get_ratio_values(h_)
            up_shift = ratio_values + ratio_err
            down_shift = ratio_values - ratio_err

            norm_factor = ratio_values.max()
            norm_factor_up = up_shift.max()
            norm_factor_down = down_shift.max()

            fit_str = f"x*0+{norm_factor:.9f}"
            fit_str_up = f"x*0+{norm_factor_up:.9f}"
            fit_str_down = f"x*0+{norm_factor_down:.9f}"

            fit_dict[era]["nominal"][(njet, njet + 1)] = [(-inf, inf, fit_str)]
            fit_dict[era]["up"][(njet, njet + 1)] = [(-inf, inf, fit_str_up)]
            fit_dict[era]["down"][(njet, njet + 1)] = [(-inf, inf, fit_str_down)]

            print(f"ratio_values: {ratio_values}")
            print(f"ratio_err: {ratio_err}")
            print(f"norm_factor: {norm_factor}")
            print(f"norm_factor_up: {norm_factor_up}")
            print(f"norm_factor_down: {norm_factor_down}")

        elif njet == 6:
            h_ = h[cat, ...]
            ratio_values, ratio_err, bin_centers = get_ratio_values(h_)
            up_shift = ratio_values + ratio_err
            down_shift = ratio_values - ratio_err

            n_bins_nonzero = np.sum(ratio_values != 0)
            # use the factor at njet==6 for all njet>=6 categories
            if n_bins_nonzero != 0:
                norm_factor = ratio_values[njet]
                norm_factor_up = up_shift[njet]
                norm_factor_down = down_shift[njet]
            else:
                raise ValueError("No non-zero bins found in ratio_values!")

            fit_str = f"x*0+{norm_factor:.9f}"
            fit_str_up = f"x*0+{norm_factor_up:.9f}"
            fit_str_down = f"x*0+{norm_factor_down:.9f}"

            fit_dict[era]["nominal"][(njet, 50)] = [(-inf, inf, fit_str)]
            fit_dict[era]["up"][(njet, 50)] = [(-inf, inf, fit_str_up)]
            fit_dict[era]["down"][(njet, 50)] = [(-inf, inf, fit_str_down)]

            print(f"ratio_values: {ratio_values}")
            print(f"ratio_err: {ratio_err}")
            print(f"n_bins_nonzero: {n_bins_nonzero}")
            print(f"norm_factor: {norm_factor}")
            print(f"norm_factor_up: {norm_factor_up}")
            print(f"norm_factor_down: {norm_factor_down}")

        else:
            continue

    print("----------")
    print(fit_dict)
    return fit_dict
