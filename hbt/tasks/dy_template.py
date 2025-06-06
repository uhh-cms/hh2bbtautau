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
                hist_producer="no_dy_weight",  # produces histograms without DY weights
                categories=("mumu__dy__os",),  # category where we want to do the fit
                variables=("njets-dilep_pt",),  # variables we want to use in the fit
            )
            for config in self.configs
        }

    def output(self):
        return self.target("hbt_corrections.json.gz")

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
    from scipy import optimize, special
    era = f"{task.config_inst.campaign.x.year}{task.config_inst.campaign.x.postfix}"

    # define transition function
    def window(x, r, s):
        """
        x: dependent variable (i.g., dilep_pt)
        r: boundary value to switch between Guassian and linear fit
        s: sign of erf function (-1 to active low pT fit function, +1 to active high pT fit function)
        """
        # the parameter 0.1 is a scaling factor to control the steepness in the transition region
        # adjust as needed dependending on the range of variable x
        return 0.5 * (special.erf(s * 0.1 * (x - r)) + 1)

    # define the fit function
    def fit_function(x, c, n, mu, sigma, a, b, r):

        """
        A fit function.
        x: dependent variable (i.g., dilep_pt)
        c: Gaussian offset
        n: Gaussian normalization
        mu and sigma: Gaussian parameters
        a and b: slope parameters
        r: boundary value to switch between Guassian and linear fit
        """

        # we choose to use a Gaussian function for low pT and a linear function for high pT
        gauss = c + (n * (1 / sigma) * np.exp(-0.5 * ((x - mu) / sigma) ** 2))
        pol = a + b * x

        # window function defines a smooth transition between the two functions at point r, which is also left floating
        return window(x, r, -1) * gauss + window(x, r, 1) * pol

    # perform the fit and save the fit function expression as a string
    def get_fit_str(njet: int, h: hist.Hist) -> dict:
        if not isinstance(njet, int):
            raise ValueError("Invalid njets value, expected int.")

        # slice the histogram for the selected njet bin
        # e.g. njet=2 will result in a histogram slice h[:, hist.loc(1.5:2.5), ...] where only the bin njets=2 survives
        jet_tupple = (njet - 0.5, njet + 0.5)
        h = h[:, hist.loc(jet_tupple[0]):hist.loc(jet_tupple[1]), ...]

        # get individual process histograms
        dy_names = [name for name in h.axes["process"] if name.startswith("dy")]
        data_names = [name for name in h.axes["process"] if name.startswith("data")]
        mc_names = [name for name in h.axes["process"] if not name.startswith(("dy", "data"))]

        # sum over processes. e.g. sum all non-DY Monte Carlo
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

        # perform the fit using our choosen fit function and given the data points in ratio_values
        param, _ = optimize.curve_fit(
            fit_function,
            bin_centers,
            ratio_values,
            p0=starting_values, method="trf",
            sigma=np.maximum(ratio_err, 1e-5),
            absolute_sigma=True,
            bounds=(lower_bounds, upper_bounds),
        )

        # read post fit parameters
        c, n, mu, sigma, a, b, r = param

        # build full string expression for the fit function
        # note: no spaces are allowed in the string. some functions are accepted like exp
        # but other function are not accepted like special.erf
        gaussian_str = f"{c}+({n}*(1/{sigma})*exp(-0.5*((x-({mu}))/{sigma})**2))"
        pol_str = f"{a}+({b})*x"
        window_str_pos = f"0.5*(special.erf(0.1*(x-{r}))+1)"  # TODO: replace special.erf string ...
        window_str_neg = f"0.5*(special.erf(-0.1*(x-{r}))+1)"  # TODO: replace special.erf string ...
        full_fit_str = f"({window_str_neg})*({gaussian_str})+({window_str_pos})*({pol_str})"

        # return function string to be passed to the json file
        return full_fit_str

    # initialize fit dictionary
    fit_dict = {
        era: {
            "nominal": {},
            # "up": {},
            # "down": {},
        },
    }

    # chose njet bin TODO: make this dynamic by reading it from event info
    njet = 2
    # build fit function string expression
    fit_str = get_fit_str(njet, h)
    # add the string expression to the dictionary, for the respective era, shift and njet bin
    # e.g. (njet, njet + 1) = (2, 3) means that the fit is appled only for njet=2
    fit_dict[era]["nominal"][(njet, njet + 1)] = [(0.0, inf, fit_str)]

    return fit_dict
