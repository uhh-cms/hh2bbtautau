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
        if not re.match(r"^(ee|mumu)__.*__os$", self.categories[0]):
            raise ValueError(f"category must start with '{{ee,mumu}}__' and end in '__os', got {self.categories[0]}")
        self.category_inst = self.config_inst.get_category(self.categories[0])

        # only one variable is allowed
        if len(self.variables) != 1:
            raise ValueError(f"{self.task_family} requires exactly one variable, got {self.variables}")
        self.variable = self.variables[0]
        # for now, variable must be "njets-dilep_pt"
        if self.variable != "njets-dilep_pt":
            raise ValueError(f"variable must be 'njets-dilep_pt', got {self.variable}")

    def output(self):
        return self.target(f"dy_weight_data__{self.category_inst.name}.pkl")

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

    def requires(self):
        return {
            config: ComuteDYWeights.req(
                self,
                config=config,
                processes=("sm_nlo_data_bkg",),  # supports groups
                hist_producer="no_dy_weight",
                categories=("mumu__dy__os",),
                variables=("njets-dilep_pt",),
            )
            for config in self.configs
        }

    def output(self):
        return self.target("hbt_corrections_incl.json.gz")

    def run(self):
        import correctionlib.schemav2 as cs
        from hbt.studies.dy_weights.create_clib_file import create_dy_weight_correction

        # load all weight data per config and merge them into a single dictionary
        dy_weight_data = law.util.merge_dicts(*(
            inp.load(formatter="pickle") for inp in self.input().values()
        ))

        # create and save the correction set
        cset = cs.CorrectionSet(
            schema_version=2,
            description="Corrections derived for the hh2bbtautau analysis.",
            corrections=[create_dy_weight_correction(dy_weight_data)],
        )
        with self.output().localize("w") as outp:
            outp.path += ".json.gz"
            with gzip.open(outp.abspath, "wt") as f:
                f.write(cset.json(exclude_unset=True))

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
    from scipy import stats, optimize, special
    from matplotlib import pyplot as plt
    era = f"{task.config_inst.campaign.x.year}{task.config_inst.campaign.x.postfix}"

    def window(x, r, s):
        """
        x: dependent variable (i.g., dilep_pt)
        r: regime boundary between two fit functions
        s: sign of erf function (+1 to active second fit function, -1 to active first fit function)
        """
        steepness = 0.1  # steepness of the erf function
        return 0.5 * (special.erf(s * steepness * (x - r)) + 1)

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
        # r = 30  # regime boundary between Guassian and linear fits

        gauss = c + (n * (1 / sigma) * np.exp(-0.5 * ((x - mu) / sigma) ** 2))
        pol = a + b * x

        # return it later as two seperate strings
        return window(x, r, -1) * gauss + window(x, r, 1) * pol

    # loop over njets bounds
    # njets = [2,3,4,5,6,7,8,9,10]
    # for nj in njets:
    #    jet_tupple = (nj - 0.5, nj + 0.5)
    #

    # h = h[:, hist.loc(jet_tupple[0]):hist.loc(jet_tupple[1]), ...]

    njet = "incl"

    if njet == "incl":
        h = h
    if njet == "2j":
        h = h[:, hist.loc(1.5):hist.loc(2.5), ...]
    elif njet == "3j":
        h = h[:, hist.loc(2.5):hist.loc(3.5), ...]
    elif njet == ">=4j":
        h = h[:, hist.loc(3.5):, ...]
    elif njet == ">=7j":
        h = h[:, hist.loc(6.5):, ...]

    dy_names = [name for name in h.axes["process"] if name.startswith("dy")]
    data_names = [name for name in h.axes["process"] if name.startswith("data")]
    mc_names = [name for name in h.axes["process"] if not name.startswith(("dy", "data"))]

    dy_h = h[{"process": list(map(hist.loc, dy_names))}][{"process": sum}]
    data_h = h[{"process": list(map(hist.loc, data_names))}][{"process": sum}]
    mc_h = h[{"process": list(map(hist.loc, mc_names))}][{"process": sum}]

    # get values and errors, and sum over njets bins
    dy_values = (dy_h.view().value).sum(axis=0)
    data_values = (data_h.view().value).sum(axis=0)
    mc_values = (mc_h.view().value).sum(axis=0)
    dy_err = (dy_h.view().variance**0.5).sum(axis=0)
    data_err = (data_h.view().variance**0.5).sum(axis=0)
    mc_err = (mc_h.view().variance**0.5).sum(axis=0)

    # print(dy_values)
    # print("dy, data and mc statistical ERRORS")
    # print(dy_err)
    # print(data_err)
    # print(mc_err)

    ratio_values = (data_values - mc_values) / dy_values
    ratio_err = (1 / dy_values) * np.sqrt(data_err**2 + mc_err**2 + (ratio_values * dy_err)**2)

    print("ratio values:")
    print(ratio_values)
    # print(ratio_err)

    # fill nans/infs and negative errors with 0
    ratio_values = np.nan_to_num(ratio_values, nan=0.0)
    ratio_values = np.where(np.isinf(ratio_values), 0.0, ratio_values)
    ratio_values = np.where(ratio_values < 0, 0.0, ratio_values)
    ratio_err = np.nan_to_num(ratio_err, nan=0.0)
    ratio_err = np.where(np.isinf(ratio_err), 0.0, ratio_err)
    ratio_err = np.where(ratio_err < 0, 0.0, ratio_err)

    bin_centers = dy_h.axes[-1].centers

    # define starting values for c, n, mu, sigma, a, b, r with respective bounds
    starting_values = [1, 1, 10, 3, 1, 0, 50]
    lower_bounds = [0.8, 0, 0, 0, 0, -1, 0]
    upper_bounds = [1.2, 10, 50, 20, 2, 2, 60]

    try:
        # perform the fit
        param, _ = optimize.curve_fit(
            fit_function,
            bin_centers,
            ratio_values,
            p0=starting_values, method="trf",
            sigma=np.maximum(ratio_err, 1e-5),
            absolute_sigma=True,
            bounds=(lower_bounds, upper_bounds),
        )
    except:
        from IPython import embed
        embed(header="Fit failed ! Start debuging ... ")

    s = np.linspace(0, 200, 1000)
    y = [fit_function(v, *param) for v in s]

    c, n, mu, sigma, a, b, r = param

    # build string expressions for the fit function
    gaussian_str = f"{c}+({n}*(1/{sigma})*exp(-0.5*((x-{mu})/{sigma})**2))"
    pol_str = f"{a}+{b}*x"

    rounded = [round(x, 4) for x in [c, n, mu, sigma, a, b, r]]
    # for param in rounded:
        # print(param)

    """
    fig, ax = plt.subplots()
    ax.plot(s, y, color="grey")
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
    ax.set_xlabel(r"$p_{T,ll}\;[\mathrm{GeV}]$")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0.4, 1.5)
    ax.set_title(f"Fit function for NLO 2022preEE, njets {njet}")
    ax.grid(True)
    fig.savefig(f"plot_{njet}.pdf")
    """

    return {
        era: {
            "nominal": {
                (0, 500): [
                    (0.0, r, gaussian_str),
                    (r, inf, pol_str),
                ],
            },
            # """
            # "up": {
            #     (0, 1): [
            #         (0.0, inf, "1.05"),
            #     ],
            #     (1, 11): [
            #         (0.0, inf, "1.05"),
            #     ],
            # },
            # "down": {
            #     (0, 1): [
            #         (0.0, inf, "0.95"),
            #     ],
            #     (1, 11): [
            #         (0.0, inf, "0.95"),
            #     ],
            # },
            # """
        },
    }
