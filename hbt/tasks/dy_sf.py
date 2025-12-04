# coding: utf-8

"""
Tasks to create correction_lib file for scale factor calculation for DY events.
"""

from __future__ import annotations

import law
import order as od
import awkward as ak
import dataclasses
import gzip
import functools
import numpy as np
import correctionlib.schemav2 as cs

from scipy import optimize
from matplotlib import pyplot as plt
from typing import Callable

from columnflow.hist_util import create_hist_from_variables, fill_hist
from hbt.tasks.base import HBTTask
from columnflow.tasks.framework.base import TaskShifts, ConfigTask
from columnflow.tasks.production import ProduceColumns
from columnflow.tasks.reduction import ProvideReducedEvents
from columnflow.util import maybe_import
from columnflow.columnar_util import (
    ChunkedIOHandler, RouteFilter, update_ak_array, attach_coffea_behavior, layout_ak_array,
    set_ak_column,
)
from columnflow.tasks.framework.mixins import (
    DatasetsProcessesMixin, ProducerClassesMixin, CalibratorClassesMixin,
    SelectorClassMixin, ReducerClassMixin,
)

hist = maybe_import("hist")


@dataclasses.dataclass
class Norm:
    nom: float
    unc: float

    @property
    def up(self) -> float:
        return self.nom + self.unc

    @property
    def down(self) -> float:
        return max(0.0, self.nom - self.unc)


class DYBaseTask(
    HBTTask,
    CalibratorClassesMixin,
    SelectorClassMixin,
    ReducerClassMixin,
    ProducerClassesMixin,
    DatasetsProcessesMixin,
):
    """
    Base class for DY tasks.
    """

    single_config = True
    allow_empty_processes = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # some useful definitions for later use
        cat_dyc = self.config_inst.get_category("dyc")
        cat_os = self.config_inst.get_category("os")
        self.category_ids = [
            cat.id for cat in cat_dyc.get_leaf_categories()
            if cat_os.has_category(cat, deep=True)
        ]

        self.dilep_pt_inst = self.config_inst.variables.n.dilep_pt
        self.nbjets_inst = self.config_inst.variables.n.nbjets_pnet_overflow
        self.njets_inst = self.config_inst.variables.n.njets

        self.variables = [
            (self.dilep_pt_inst, "dilep_pt"),
            (self.nbjets_inst, "nbjets"),
        ]
        self.variables_names = [var_name for _, var_name in self.variables]

    @classmethod
    def modify_param_values(cls, params):
        params = super().modify_param_values(params)
        params["processes"] = tuple()
        return params

    @classmethod
    def resolve_param_values(cls, params):
        params["known_shifts"] = TaskShifts()
        return super().resolve_param_values(params)

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "datasets", f"datasets_{self.datasets_repr}")
        return parts


class LoadDYData(DYBaseTask):
    """
    Example command:

        > law run hbt.LoadDYData \
            --config 22pre_v14 \
            --version prod20_vbf
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.read_columns = [
            "Jet.btagPNetB",
            "channel_id",
            "category_ids",
            "process_id",
            "Electron.pt",
            "Electron.eta",
            "Electron.phi",
            "Electron.mass",
            "Muon.mass",
            "Muon.phi",
            "Muon.eta",
            "Muon.pt",
            "Tau.mass",
            "Tau.phi",
            "Tau.eta",
            "Tau.pt",
            "gen_dilepton_pt",
        ]
        self.event_weight_columns = [
            "normalization_weight",
            "normalized_pdf_weight",
            "normalized_murmuf_weight",
            "normalized_pu_weight",
            "normalized_isr_weight",
            "normalized_fsr_weight",
            "normalized_njet_btag_weight_pnet",
            "electron_id_weight",
            "electron_reco_weight",
            "muon_id_weight",
            "muon_iso_weight",
            "tau_weight",
            "trigger_weight",
        ]
        self.dataset_event_weight_columns = {
            "tt_*": ["top_pt_weight"],
        }
        self.write_columns = [
            "channel_id",
            "category_ids",
            "process_id",
            "dilep_pt",
            "gen_dilepton_pt",
            "weight",
            "njets",
            "nbjets",
        ]

    def output(self):
        return self.target("data.pkl")

    def requires(self):
        reqs = {}
        for dataset in self.datasets:
            reqs[dataset] = {
                "reduction": ProvideReducedEvents.req(self, dataset=dataset),
                "production": {
                    prod: ProduceColumns.req(self, dataset=dataset, producer=prod)
                    for prod in self.producers
                },
            }

        return reqs

    def run(self):
        outputs = self.output()

        data_events = []
        dy_events = []
        bkg_events = []

        # loop over datasets and load inputs
        for dataset_name, inps in self.input().items():
            self.publish_message(f"Loading dataset '{dataset_name}'")

            # prepare columns to write
            route_filter = RouteFilter(keep=self.write_columns)

            # define columns to read
            read_columns = [
                *self.read_columns,
                *self.event_weight_columns,
            ]
            dataset_weight_columns = []
            for pattern, cols in self.dataset_event_weight_columns.items():
                if law.util.multi_match(dataset_name, pattern):
                    dataset_weight_columns += cols
            read_columns += dataset_weight_columns

            # loop over each file per input
            coll = inps["reduction"].collection
            for i in range(len(coll)):
                targets = [coll.targets[i]["events"]]
                for prod in self.producers:
                    targets.append(inps["production"][prod].collection.targets[i]["columns"])

                # prepare inputs for localization
                with law.localize_file_targets(targets, mode="r") as local_inps:
                    reader = ChunkedIOHandler(
                        [t.abspath for t in local_inps],
                        source_type=len(targets) * ["awkward_parquet"],
                        read_columns=len(targets) * [read_columns],
                        chunk_size=50_000,
                    )
                    for (events, *columns), pos in reader:
                        events = update_ak_array(events, *columns)
                        events = attach_coffea_behavior(events)

                        # filter events for DY weight derivation
                        cat_mask = np.isin(ak.flatten(events.category_ids), self.category_ids)
                        cat_mask = layout_ak_array(cat_mask, events.category_ids)
                        event_mask = (
                            ak.any(cat_mask, axis=1) &
                            (
                                (events.channel_id == self.config_inst.channels.n.ee.id) |
                                (events.channel_id == self.config_inst.channels.n.mumu.id)
                            )
                        )
                        events = events[event_mask]

                        # compute additional columns
                        events = set_ak_column(
                            events,
                            "njets",
                            ak.num(events.Jet, axis=1),
                            value_type=np.int32,
                        )

                        wp_value = self.config_inst.x.btag_working_points.particleNet.medium
                        bjet_mask = events.Jet.btagPNetB >= wp_value
                        events = set_ak_column(
                            events,
                            "nbjets",
                            ak.sum(bjet_mask, axis=1),
                            value_type=np.int32,
                        )

                        events = set_ak_column(
                            events,
                            "dilep_pt",
                            self.dilep_pt_inst.expression(events),
                        )

                        weight = np.ones(len(events), dtype=np.float32)
                        if not dataset_name.startswith("data_"):
                            for col in self.event_weight_columns + dataset_weight_columns:
                                if col in events.fields:
                                    weight = weight * events[col]
                        events = set_ak_column(events, "weight", weight)

                        # filter columns to read at the end
                        events = route_filter(events)
                        events.behavior = None

                        # save events by dataset type
                        if dataset_name.startswith("data_"):
                            data_events.append(events)
                        elif dataset_name.startswith("dy_"):
                            dy_events.append(events)
                        else:
                            bkg_events.append(events)

        data_events = ak.concatenate(data_events, axis=0) if data_events else None
        dy_events = ak.concatenate(dy_events, axis=0) if dy_events else None
        bkg_events = ak.concatenate(bkg_events, axis=0) if bkg_events else None

        outputs.dump((data_events, dy_events, bkg_events), formatter="pickle")


class DYWeights(DYBaseTask):
    """
    Example command:

        > law run hbt.DYWeights \
            --config 22pre_v14 \
            --datasets bkg_data \
            --version prod20_vbf
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # define possible uncertainty factors for fit shifts
        self.unc_factors = [1.5]

        self.fit_identifiers = ["fit_njets2", "fit_njets3", "fit_njets4"]

    def output(self):
        outputs = {
            "plots": {},
        }

        for factor in self.unc_factors:
            outputs["weights"] = self.target("weights.pkl")

        for tmp_id in self.fit_identifiers:
            for factor in self.unc_factors:
                tmp_id_full = f"{tmp_id}_unc{factor}"
                outputs["plots"][tmp_id_full] = self.target(f"{tmp_id_full}.pdf")

        return outputs

    def requires(self):
        return LoadDYData.req(self)

    def run(self):
        outputs = self.output()

        # read data, potentially from cache
        data_events, dy_events, bkg_events = self.input().load(formatter="pickle")

        @dataclasses.dataclass
        class Deps:
            fit_njet_bin: tuple[int, int]
            fit_syst: str

        @dataclasses.dataclass
        class FitResult:
            norm_value: float | str
            fit_string: str

            def to_window(self):
                return [(0.0, float("inf"), f"{self.norm_value}*({self.fit_string})")]

        dict_setup = {
            "nominal": (create_setup := lambda fit_syst: ({
                (2, 3): {
                    (0, 1): Deps(fit_njet_bin=(2, 3), fit_syst=fit_syst),
                    (1, 2): Deps(fit_njet_bin=(2, 3), fit_syst=fit_syst),
                    (2, 101): Deps(fit_njet_bin=(2, 3), fit_syst=fit_syst),
                },
                (3, 4): {
                    (0, 1): Deps(fit_njet_bin=(3, 4), fit_syst=fit_syst),
                    (1, 2): Deps(fit_njet_bin=(3, 4), fit_syst=fit_syst),
                    (2, 101): Deps(fit_njet_bin=(3, 4), fit_syst=fit_syst),
                },
                (4, 5): {
                    (0, 1): Deps(fit_njet_bin=(4, 101), fit_syst=fit_syst),
                    (1, 2): Deps(fit_njet_bin=(4, 101), fit_syst=fit_syst),
                    (2, 101): Deps(fit_njet_bin=(4, 101), fit_syst=fit_syst),
                },
                (5, 6): {
                    (0, 1): Deps(fit_njet_bin=(4, 101), fit_syst=fit_syst),
                    (1, 2): Deps(fit_njet_bin=(4, 101), fit_syst=fit_syst),
                    (2, 101): Deps(fit_njet_bin=(4, 101), fit_syst=fit_syst),
                },
                (6, 101): {
                    (0, 1): Deps(fit_njet_bin=(4, 101), fit_syst=fit_syst),
                    (1, 2): Deps(fit_njet_bin=(4, 101), fit_syst=fit_syst),
                    (2, 101): Deps(fit_njet_bin=(4, 101), fit_syst=fit_syst),
                },
            }))("nominal"),
            "syst_gauss_up": create_setup("syst_gauss_up"),
            "syst_gauss_down": create_setup("syst_gauss_down"),
            "syst_linear_up": create_setup("syst_linear_up"),
            "syst_linear_down": create_setup("syst_linear_down"),
            "syst_up": create_setup("syst_up"),
            "syst_down": create_setup("syst_down"),
            "stat_btag0_up": create_setup("nominal"),
            "stat_btag0_down": create_setup("nominal"),
            "stat_btag1_up": create_setup("nominal"),
            "stat_btag1_down": create_setup("nominal"),
            "stat_btag2_up": create_setup("nominal"),
            "stat_btag2_down": create_setup("nominal"),
            "stat_up": create_setup("nominal"),
            "stat_down": create_setup("nominal"),
        }

        # get dict_out for each possible uncertainty factor
        for factor in self.unc_factors:
            print("-> using unc factor:", factor)
            # initialize dicts to be updated
            dict_out = {}
            ratios = {}
            fit_params = {}

            @functools.cache
            def get_fit(fit_njet_bin, fit_syst, factor) -> tuple[Callable, str, tuple[float, ...]]:
                var = self.dilep_pt_inst

                data_mask = self.get_mask(data_events, njet_bin=fit_njet_bin, channel="mumu")
                data_hist = self.hist_function(var, data_events[var.name][data_mask], data_events.weight[data_mask])

                dy_mask = self.get_mask(dy_events, njet_bin=fit_njet_bin, channel="mumu")
                dy_hist = self.hist_function(var, dy_events[var.name][dy_mask], dy_events.weight[dy_mask])

                bkg_mask = self.get_mask(bkg_events, njet_bin=fit_njet_bin, channel="mumu")
                bkg_hist = self.hist_function(var, bkg_events[var.name][bkg_mask], bkg_events.weight[bkg_mask])

                ratio_values, ratio_err, bin_centers = self.get_ratio_values(
                    data_hist,
                    dy_hist,
                    bkg_hist,
                    var,
                )

                # cache ratio values for later plotting
                if fit_syst == "nominal":
                    ratios[fit_njet_bin] = (ratio_values, ratio_err, bin_centers)

                # change depending on fit_syst
                if fit_syst.startswith("syst_"):

                    nominal_r = get_fit(fit_njet_bin, "nominal", factor)[2][6]  # get nominal r value

                    if fit_syst in ["syst_up", "syst_down"]:
                        bin_mask = np.ones_like(bin_centers, dtype=bool)
                    else:
                        bin_mask = bin_centers <= nominal_r
                        bin_mask = bin_mask if "gauss" in fit_syst else ~bin_mask

                    sign = 1.0 if "up" in fit_syst else -1.0
                    ratio_values = np.where(bin_mask, ratio_values + sign * factor * ratio_err, ratio_values)

                # define starting values with respective bounds
                starting_values = [1, 1, 10, 3, 1, 0, 50]
                lower_bounds = [0.6, 0, 0, 0, 0, -2, 20]
                upper_bounds = [1.2, 10, 50, 20, 2, 3, 100]

                # perform the fit
                popt, pcov = optimize.curve_fit(
                    self.get_fit_function,
                    bin_centers,
                    ratio_values,
                    p0=starting_values, method="trf",
                    sigma=np.maximum(ratio_err, 1e-5),
                    absolute_sigma=True,
                    bounds=(lower_bounds, upper_bounds),
                )

                c, n, mu, sigma, a, b, r = popt
                fit_str = self.get_fit_str(*popt)

                # placeholder for fit calculation
                return functools.partial(self.get_fit_function, c=c, n=n, mu=mu, sigma=sigma, a=a, b=b, r=r), fit_str, popt  # noqa: E501

            @functools.cache
            def get_norm(njet_bin, nbjet_bin, syst, fit_njet_bin, fit_syst) -> float:

                var = self.nbjets_inst
                var.name = "nbjets"

                data_mask = self.get_mask(data_events, njet_bin=njet_bin, nbjet_bin=nbjet_bin, channel="mumu")
                data_hist = self.hist_function(var, data_events[var.name][data_mask], data_events.weight[data_mask])

                dy_mask = self.get_mask(dy_events, njet_bin=njet_bin, nbjet_bin=nbjet_bin, channel="mumu")
                fit_funct = get_fit(fit_njet_bin, fit_syst, factor)[0]
                dy_weight = dy_events.weight[dy_mask] * fit_funct(dy_events.gen_dilepton_pt[dy_mask])
                dy_hist = self.hist_function(var, dy_events[var.name][dy_mask], dy_weight)

                bkg_mask = self.get_mask(bkg_events, njet_bin=njet_bin, nbjet_bin=nbjet_bin, channel="mumu")
                bkg_hist = self.hist_function(var, bkg_events[var.name][bkg_mask], bkg_events.weight[bkg_mask])

                ratio_values, ratio_err, bin_centers = self.get_ratio_values(
                    data_hist,
                    dy_hist,
                    bkg_hist,
                    var,
                )

                norm = Norm(ratio_values[nbjet_bin[0]], ratio_err[nbjet_bin[0]])
                norm_value = norm.nom

                # general syst case
                if syst in ["stat_up", "stat_down"]:
                    norm_value = norm.up if syst.endswith("up") else norm.down
                # nbjet syst cases
                if nbjet_bin[0] == 0 and syst.startswith("stat_btag0_"):
                    norm_value = norm.up if syst.endswith("up") else norm.down
                elif nbjet_bin[0] == 1 and syst.startswith("stat_btag1_"):
                    norm_value = norm.up if syst.endswith("up") else norm.down
                elif nbjet_bin[0] == 2 and syst.startswith("stat_btag2_"):
                    norm_value = norm.up if syst.endswith("up") else norm.down

                return norm_value

            for syst in dict_setup.keys():
                dict_out[syst] = {}
                for njet_bin in dict_setup[syst]:
                    dict_out[syst][njet_bin] = {}
                    for nbjet_bin, deps in dict_setup[syst][njet_bin].items():

                        fit_function, fit_string, fit_popt = get_fit(deps.fit_njet_bin, deps.fit_syst, factor)
                        norm_value = get_norm(njet_bin, nbjet_bin, syst, deps.fit_njet_bin, deps.fit_syst)

                        # ending up in fit result
                        fit_result = FitResult(norm_value, fit_string)

                        # save fit parameters for later plotting
                        if deps.fit_syst not in fit_params.keys():
                            fit_params[deps.fit_syst] = {}
                        fit_params[deps.fit_syst][deps.fit_njet_bin] = fit_popt

                        # store it
                        dict_out[syst][njet_bin][nbjet_bin] = fit_result.to_window()

            # create and save fit plot with systematic uncertainty bands
            for fit_njet_bins, values in ratios.items():
                ratio_values, ratio_err, bin_centers = values
                self.get_fit_plot(fit_njet_bins, fit_params, factor, ratio_values, ratio_err, bin_centers)

            # save final dy weights
            outputs["weights"].dump(dict_out, formatter="pickle")

    def get_mask(self, events, njet_bin=None, nbjet_bin=None, channel=None):
        mask = np.ones(len(events), dtype=bool)
        if njet_bin is not None:
            njet_min, njet_max = njet_bin
            mask = mask & ((events.njets >= njet_min) & (events.njets < njet_max))
        if nbjet_bin is not None:
            nbjet_min, nbjet_max = nbjet_bin
            mask = mask & ((events.nbjets >= nbjet_min) & (events.nbjets < nbjet_max))
        if channel == "ee":
            mask = mask & (events.channel_id == self.config_inst.channels.n.ee.id)
        elif channel == "mumu":
            mask = mask & (events.channel_id == self.config_inst.channels.n.mumu.id)
        return mask

    def hist_function(self, var, data, weights):
        h = create_hist_from_variables(var, weight=True)
        fill_hist(h, {var.name: data, "weight": weights})
        return h

    def get_ratio_values(
            self,
            data_h: hist.Hist,
            dy_h: hist.Hist,
            bkg_h: hist.Hist,
            variable_inst: od.Variable,
    ) -> tuple[hist.Hist, hist.Hist, hist.Hist]:

        # under/overflow treatment
        def fix_flow(h: hist.Hist):
            h = h.copy()
            if variable_inst.x("underflow", False):
                v = h.view(flow=True)
                v.value[..., 1] += v.value[..., 0]
                v.variance[..., 1] += v.variance[..., 0]
                v.value[..., 0] = 0.0
                v.variance[..., 0] = 0.0
            if variable_inst.x("overflow", False):
                v = h.view(flow=True)
                v.value[..., -2] += v.value[..., -1]
                v.variance[..., -2] += v.variance[..., -1]
                v.value[..., -1] = 0.0
                v.variance[..., -1] = 0.0
            return h

        data_h = fix_flow(data_h)
        dy_h = fix_flow(dy_h)
        bkg_h = fix_flow(bkg_h)

        # get bin centers
        bin_centers = dy_h.axes[-1].centers

        # get histogram values and errors
        data_values = data_h.view().value
        data_err = data_h.view().variance**0.5

        dy_values = dy_h.view().value
        dy_err = dy_h.view().variance**0.5

        bkg_values = bkg_h.view().value
        bkg_err = bkg_h.view().variance**0.5

        # calculate (data-bkg)/dy ratio factor with statistical error
        ratio_values = (data_values - bkg_values) / dy_values
        ratio_err = (1 / dy_values) * np.sqrt(data_err**2 + bkg_err**2 + (ratio_values * dy_err)**2)

        # fill nans/infs and negative errors with 0.0
        ratio_values = np.nan_to_num(ratio_values, nan=0.0)
        ratio_values = np.where(np.isinf(ratio_values), 0.0, ratio_values)
        ratio_values = np.where(ratio_values < 0, 0.0, ratio_values)
        ratio_err = np.nan_to_num(ratio_err, nan=0.0)
        ratio_err = np.where(np.isinf(ratio_err), 0.0, ratio_err)
        ratio_err = np.where(ratio_err < 0, 0.0, ratio_err)

        return (ratio_values, ratio_err, bin_centers)

    def get_fit_function(self, x, c, n, mu, sigma, a, b, r):

        from scipy import special

        """
        x: dependent variable (i.g., dilep_pt)
        c: Gaussian offset
        n: Gaussian normalization
        mu and sigma: Gaussian parameters
        a, b: polinomial parameters
        r: regime boundary between Guassian and linear fits
        """

        # choose gaussian and linear functions to do the fit
        # we cap x at 200 GeV to avoid falling off the function range
        gauss = c + (n * (1 / sigma) * np.exp(-0.5 * ((np.minimum(x, 200) - mu) / sigma) ** 2))
        pol = a + b * np.minimum(x, 200)

        # parameter to control the transition smoothness between the two functions
        step_par = 0.08

        # use scipy erf function to create smooth transition
        sci_erf_neg = (0.5 * (special.erf(-step_par * (np.minimum(x, 200) - r)) + 1))
        sci_erf_pos = (0.5 * (special.erf(step_par * (np.minimum(x, 200) - r)) + 1))

        return sci_erf_neg * gauss + sci_erf_pos * pol

    def get_fit_str(self, c, n, mu, sigma, a, b, r):
        # build fit function string
        gauss = f"(({c})+(({n})*(1/{sigma})*exp(-0.5*((min(x,200)-{mu})/{sigma})^2)))"
        pol = f"(({a})+({b})*min(x,200))"
        fit_string = f"(0.5*(erf(-0.08*(min(x,200)-{r}))+1))*{gauss}+(0.5*(erf(0.08*(min(x,200)-{r}))+1))*{pol}"

        return fit_string

    def get_fit_plot(self, fit_njet_bin, fit_params, factor, ratio_values, ratio_err, bin_centers):
        outputs = self.output()

        # initialize figure
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.subplots_adjust(top=0.93)

        # get nominal fit
        s = np.linspace(0, 200, 1000)
        y_nom = [self.get_fit_function(v, *fit_params["nominal"][fit_njet_bin]) for v in s]
        ax.plot(s, y_nom, color="black", label="Nominal", lw=2)

        # get shifted fits
        for regime, colour, label in [("syst_gauss", "red", "Gaussian (up/down)"), ("syst_linear", "blue", "Linear (up/down)")]:  # noqa: E501
            y_up = [self.get_fit_function(v, *fit_params[f"{regime}_up"][fit_njet_bin]) for v in s]
            y_down = [self.get_fit_function(v, *fit_params[f"{regime}_down"][fit_njet_bin]) for v in s]
            ax.fill_between(s, y_up, y_nom, color=colour, alpha=0.2, label=label)
            ax.fill_between(s, y_down, y_nom, color=colour, alpha=0.2)

        # plot ratio error bars
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

        # build labels
        njets = fit_njet_bin[0]
        tmp_label = rf"$={njets}$" if njets < 4 else rf"$\geq {njets}$"
        jet_label = rf"$\mu\mu$ channel, DY region, AK4 jets {tmp_label}"
        label = rf"Private work (CMS data/simulation)      {jet_label}"

        # styling and legends
        ax.legend(loc="lower right")
        ax.set_xlabel(r"$\mathrm{p}_{\mathrm{T,ll}} \ [\mathrm{GeV}]$", fontsize=15, loc="right")
        ax.set_ylabel("Data - MC / DY", fontsize=15, loc="top")
        ax.grid(True)
        fig.text(0.12, 0.97, label, verticalalignment="top", horizontalalignment="left", fontsize=13)
        ax.tick_params(axis="both", labelsize=15)

        # save plot
        for key in outputs["plots"].keys():
            if (f"{int(njets)}" in key) and (f"{factor}" in key):
                outputs["plots"][key].dump(fig, formatter="mpl")


class ExportDYWeights(HBTTask, ConfigTask):
    """
    Example command:

        > law run hbt.ExportDYWeights \
            --configs 22pre_v14,22post_v14,... \
            --hbt.DYWeights-datasets bkg_data \
            --version prod20_vbf
    """
    single_config = False

    def requires(self):
        return {
            config_inst: DYWeights.req(
                self,
                config=config_inst.name,
                datasets=("bkg_data",),
            )
            for config_inst in self.config_insts
        }

    def output(self):
        return self.target("hbt_corrections.json.gz")

    def run(self):
        dy_weight_data = {}

        for config_inst, inps in self.input().items():
            weight_data = inps["weights"].load(formatter="pickle")
            era = config_inst.x.dy_weight_config.era
            dy_weight_data[era] = weight_data

        cset = cs.CorrectionSet(
            schema_version=2,
            description="Corrections derived for the hh2bbtautau analysis.",
            corrections=[
                self.create_dy_weight_correction(dy_weight_data),
            ],
        )

        outp = self.output()
        outp.parent.touch()
        with gzip.open(outp.abspath, "wt") as f:
            f.write(cset.model_dump_json(exclude_unset=True))

        # validate the content
        law.util.interruptable_popen(f"correction summary {outp.abspath}", shell=True)

    def expr_in_range(self, expr: str, lower_bound: float | int, upper_bound: float | int) -> str:
        # lower bound must be smaller than upper bound
        assert lower_bound < upper_bound
        return expr

    def create_dy_weight_correction(self, dy_weight_data: dict) -> cs.Correction:
        # create the correction object
        dy_weight_correction = cs.Correction(
            name="dy_weight",
            description="DY weights derived in the phase space of the hh2bbtautau analysis, supposed to correct njet and "  # noqa: E501
            "ptll distributions, as well as correlated quantities.",
            version=1,
            inputs=[
                cs.Variable(name="era", type="string", description="Era name."),
                cs.Variable(name="njets", type="int", description="Number of jets in the event."),
                cs.Variable(name="ntags", type="int", description="Number of (PNet) b-tagged jets."),
                cs.Variable(name="ptll", type="real", description="Gen level pT of the dilepton system [GeV]."),
                cs.Variable(name="syst", type="string", description="Systematic variation."),
            ],
            output=cs.Variable(name="weight", type="real", description="DY event weight."),
            data=cs.Category(
                nodetype="category",
                input="era",
                content=[],
            ),
        )
        # dynamically fill it
        for era, era_data in dy_weight_data.items():
            era_category_content = []
            for syst, syst_data in era_data.items():
                njet_bin_content = []
                for (min_njet, max_njet), ntag_data in syst_data.items():
                    ntag_bin_content = []
                    for (min_ntag, max_ntag), formulas in ntag_data.items():
                        # create a joined expression for all formulas
                        expr = "+".join(
                            self.expr_in_range(formula, lower_bound, upper_bound)
                            for lower_bound, upper_bound, formula in formulas
                        )
                        # add formula object to binning content
                        ntag_bin_content.append(
                            cs.Formula(
                                nodetype="formula",
                                variables=["ptll"],
                                parser="TFormula",
                                expression=expr,
                            ),
                        )
                    njet_bin_content.append(
                        cs.Binning(
                            nodetype="binning",
                            input="ntags",
                            flow="error",
                            edges=sorted(set(sum(map(list, ntag_data.keys()), []))),
                            content=ntag_bin_content,
                        ),
                    )
                # add a new category item for the jet bins
                era_category_content.append(
                    cs.CategoryItem(
                        key=syst,
                        value=cs.Binning(
                            nodetype="binning",
                            input="njets",
                            flow="error",
                            edges=sorted(set(sum(map(list, syst_data.keys()), []))),
                            content=njet_bin_content,
                        ),
                    ),
                )
            # add a new category item for the era
            dy_weight_correction.data.content.append(
                cs.CategoryItem(
                    key=era,
                    value=cs.Category(
                        nodetype="category",
                        input="syst",
                        content=era_category_content,
                    ),
                ),
            )
        return dy_weight_correction
