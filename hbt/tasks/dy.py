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
            --processes sm_nlo_data_bkg \
            --version prod8_dy \
            --hist-producer no_dy_weight \
            --categories mumu__dy__os \
            --variables njets-dilep_pt
    """

    single_config = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # only one category is allowed right now
        if len(self.categories) != 1:
            raise ValueError(f"{self.task_family} requires exactly one category, got {self.categories}")
        # ensure that the category matches a specific pattern: starting with "ee"/"mumu" and ending in "os"
        if not re.match(r"^(ee|mumu|tautau)__.*__os.*$", self.categories[0]):
            raise ValueError(f"category must start with '{{ee,mumu,tautau}}__' and contain '__os', got {self.categories[0]}")
        self.category_inst = self.config_inst.get_category(self.categories[0])

        # only one variable is allowed
        if len(self.variables) != 1:
            raise ValueError(f"{self.task_family} requires exactly one variable, got {self.variables}")
        self.variable = self.variables[0]
        # for now, variable must be "njets-dilep_pt"
        if self.variable not in ["njets-dilep_pt", "njets-gen_dilepton_pt"]:
            raise ValueError(f"variable must be 'njets-dilep_pt' or 'njets-gen_dilepton_pt', got {self.variable}")

    def output(self):
        return self.target(f"dy_weight_data_{self.categories[0]}_{self.variables[0]}.pkl")

    def run(self):
        # prepare categories to sum over
        leaf_category_insts = self.category_inst.get_leaf_categories() or [self.category_inst]

        # load histograms from all input datasets and add them
        h = None
        for dataset_name, inp in self.input().items():
            h_ = inp.collection[0]["hists"][self.variable].load(formatter="pickle")

            # select and sum over leaf categories
            h_ = h_[{"category": [hist.loc(cat.name) for cat in leaf_category_insts if cat.name in h_.axes["category"]]}]
            h_ = h_[{"category": sum}]

            # use the nominal shift only
            h_ = h_[{"shift": hist.loc("nominal")}]

            # add it
            h = h_ if h is None else (h + h_)

        # compute the dy weight data
        dy_weight_data = compute_weight_data(self, h)

        # store them
        self.output().dump(dy_weight_data, formatter="pickle")


class ExportDYWeights(HBTTask, ConfigTask):
    """
    Example command:

        > law run hbt.ExportDYWeights \
            --configs 22pre_v14,22post_v14,... \
            --version prod8_dy
    """

    sandbox = dev_sandbox("bash::$CF_BASE/sandboxes/venv_columnar.sh")

    single_config = False

    category = "tautau__dyc__os__noniso"
    variable = "njets-dilep_pt"

    def requires(self):
        return {
            config: ComuteDYWeights.req(
                self,
                config=config,
                processes=("sm_nlo_data_bkg",),  # supports groups
                hist_producer="no_dy_weight",
                categories=(self.category,),
                variables=(self.variable,),
            )
            for config in self.configs
        }

    def output(self):
        return self.target(f"hbt_corrections_{self.category}_{self.variable}.json.gz")

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
    era = f"{task.config_inst.campaign.x.year}{task.config_inst.campaign.x.postfix}"

    # hardcode erf function definition
    def erf(x):
        x = np.asarray(x)
        # Constants in the approximation formula
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        # Save the sign of x
        sign = np.sign(x)
        abs_x = np.abs(x)

        # Abramowitz and Stegun approximation
        t = 1.0 / (1.0 + p * abs_x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-abs_x * abs_x)

        return sign * y

    def my_fit_function(x, c, n, mu, sigma, a, b, r):

        """
        A fit function.
        x: dependent variable (i.g., dilep_pt)
        c: Gaussian offset
        n: Gaussian normalization
        mu and sigma: Gaussian parameters
        a and b: slope parameters
        r: regime boundary between Guassian and linear fits
        """

        gauss = c + (n * (1 / sigma) * np.exp(-0.5 * ((x - mu) / sigma) ** 2))
        pol = a + b * x

        my_fit_function = (0.5 * (erf(-0.1 * (x - r)) + 1)) * gauss + (0.5 * (erf(0.1 * (x - r)) + 1)) * pol

        return my_fit_function

    # use scipy erf function definition
    def window(x, r, s):
        """
        x: dependent variable (i.g., dilep_pt)
        r: regime boundary between two fit functions
        s: sign of erf function (+1 to active second fit function, -1 to active first fit function)
        """

        sci_erf = 0.5 * (special.erf(s * 0.1 * (x - r)) + 1)

        return sci_erf

    def fit_function(x, c, n, mu, sigma, a, b, r):

        """
        A fit function.
        x: dependent variable (i.g., dilep_pt)
        c: Gaussian offset
        n: Gaussian normalization
        mu and sigma: Gaussian parameters
        a and b: slope parameters
        r: regime boundary between Guassian and linear fits
        """

        gauss = c + (n * (1 / sigma) * np.exp(-0.5 * ((x - mu) / sigma) ** 2))
        pol = a + b * x

        return window(x, r, -1) * gauss + window(x, r, 1) * pol

    # build fit function string
    def get_fit_str(njet: int, h: hist.Hist) -> dict:
        if njet not in [2, 3, 4]:
            raise ValueError(f"Invalid njets value {njet}, expected 2, 3, or 4.")

        # slice the histogram for the current njets bin
        jet_tupple = (njet - 0.5, njet + 0.5)

        # slice the histogram for the selected njets bin
        h = h[:, hist.loc(jet_tupple[0]):hist.loc(jet_tupple[1]), ...]

        # get and sum process histograms
        dy_names = [name for name in h.axes["process"] if name.startswith("dy")]
        data_names = [name for name in h.axes["process"] if name.startswith("data")]
        mc_names = [name for name in h.axes["process"] if not name.startswith(("dy", "data"))]
        dy_h = h[{"process": list(map(hist.loc, dy_names))}][{"process": sum}]
        data_h = h[{"process": list(map(hist.loc, data_names))}][{"process": sum}]
        mc_h = h[{"process": list(map(hist.loc, mc_names))}][{"process": sum}]

        # get bin centers
        bin_centers = dy_h.axes[-1].centers

        # get histogram values and errors, summing over njets axis
        dy_values = (dy_h.view().value).sum(axis=0)
        data_values = (data_h.view().value).sum(axis=0)
        mc_values = (mc_h.view().value).sum(axis=0)
        dy_err = (dy_h.view().variance**0.5).sum(axis=0)
        data_err = (data_h.view().variance**0.5).sum(axis=0)
        mc_err = (mc_h.view().variance**0.5).sum(axis=0)

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

        # define starting values for c, n, mu, sigma, a, b, r with respective bounds
        starting_values = [1, 1, 10, 3, 1, 0, 50]
        lower_bounds = [0.8, 0, 0, 0, 0, -1, 0]
        upper_bounds = [1.2, 10, 50, 20, 2, 2, 60]

        from IPython import embed; embed(header="dy fit ...")

        # perform the fit with both functions
        param_fit, _ = optimize.curve_fit(
            fit_function,
            bin_centers,
            ratio_values,
            p0=starting_values, method="trf",
            sigma=np.maximum(ratio_err, 1e-5),
            absolute_sigma=True,
            bounds=(lower_bounds, upper_bounds),
        )

        # get post fit parameter and convert to strings
        c, n, mu, sigma, a, b, r = param_fit

        c = f"{c:.9f}"
        n = f"{n:.9f}"
        mu = f"{mu:.9f}"
        sigma = f"{sigma:.9f}"
        a = f"{a:.9f}"
        b = f"{b:.9f}"
        r = f"{r:.9f}"

        # create Gaussian and polinomial strings
        gauss = f"(({c})+(({n})*(1/{sigma})*exp(-0.5*((x-{mu})/{sigma})^2)))"
        pol = f"(({a})+({b})*x)"

        # full fit function string
        fit_string = f"(0.5*(erf(-0.1*(x-{r}))+1))*{gauss}+(0.5*(erf(0.1*(x-{r}))+1))*{pol}"
        print(fit_string)
        # -------------------------------
        #  plot the fit function‚
        s = np.linspace(0, 200, 1000)
        y = [fit_function(v, *param_fit) for v in s]
        fig, ax = plt.subplots()
        ax.plot(s, y, color="grey", label="fit")
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
        ax.set_title("NNLO 2022preEE, mumu__dy_eq2j_eq0b")
        ax.grid(True)
        fig.savefig("plot_NNLO_mumu__dy_eq2j_eq0b.pdf")
        # -------------------------------
        print(fit_string)

        return fit_string

    # initialize fit dictionary
    fit_dict = {
        era: {
            "nominal": {},
            # "up": {},
            # "down": {},
        },
    }

    for njet in [2]:
        fit_str = get_fit_str(njet, h)

        # TODO: if first 2 elements in tupple are not -inf, inf the fit_str will be multiplied
        # by a step function scipy.special.erf with parameter 1000

        # temporarily use the same fit function for all njets
        fit_dict[era]["nominal"][(0, 500)] = [(-inf, inf, fit_str)]

    return fit_dict
