# coding: utf-8

"""
Tasks to create correction_lib file for scale factor calculation for DY events.
"""

from __future__ import annotations

import luigi
import law
import order as od
import awkward as ak
import dataclasses

from columnflow.tasks.framework.base import TaskShifts
from columnflow.tasks.framework.mixins import (
    DatasetsProcessesMixin, ProducerClassesMixin, CalibratorClassesMixin,
    SelectorClassMixin, ReducerClassMixin,
)
from columnflow.tasks.production import ProduceColumns
from columnflow.tasks.reduction import ProvideReducedEvents
from columnflow.util import maybe_import
from columnflow.columnar_util import (
    ChunkedIOHandler, RouteFilter, update_ak_array, attach_coffea_behavior, layout_ak_array,
    set_ak_column,
)
from columnflow.hist_util import create_hist_from_variables, fill_hist

from hbt.tasks.base import HBTTask

import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline

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


class DYWeights(
    HBTTask,
    CalibratorClassesMixin,
    SelectorClassMixin,
    ReducerClassMixin,
    ProducerClassesMixin,
    DatasetsProcessesMixin
):
    """
    some description
    """

    single_config = True
    allow_empty_processes = True

    reload = luigi.BoolParameter(default=False, significant=False)

    @classmethod
    def modify_param_values(cls, params):
        params = super().modify_param_values(params)
        params["processes"] = tuple()
        return params

    @classmethod
    def resolve_param_values(cls, params):
        params["known_shifts"] = TaskShifts()
        return super().resolve_param_values(params)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # get era
        era = self.config_inst.aux.get('dy_weight_config').era

        # initialize dictionary to store results
        dy_weight_data = {}

        # get variable instances
        self.dilep_pt_inst = self.config_inst.variables.n.dilep_pt
        self.nbjets_inst = self.config_inst.variables.n.nbjets_pnet_overflow
        self.njets_inst = self.config_inst.variables.n.njets

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
            "gen_dilep_pt",
        ]
        self.event_weight_columns = [
            "normalization_weight",
            "normalized_pdf_weight",
            "normalized_murmuf_weight",
            "normalized_pu_weight",
            "normalized_isr_weight",
            "normalized_fsr_weight",
            "normalized_njet_btag_weight_pnet",
            "electron_weight",
            "muon_weight",
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
            "gen_dilep_pt",
            "weight",
            "njets",
            "nbjets",
        ]

        cat_dyc = self.config_inst.get_category("dyc")
        cat_os = self.config_inst.get_category("os")
        self.category_ids = [
            cat.id for cat in cat_dyc.get_leaf_categories()
            if cat_os.has_category(cat, deep=True)
        ]

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "datasets", f"datasets_{self.datasets_repr}")
        return parts

    def output(self):
        outputs = {
            "plots": [],
            "json": self.target("corrections.json.gz"),
            "data": self.target("data.pkl", optional=True),
        }

        return outputs

    def requires(self):
        if not self.reload and self.output()["data"].exists():
            return []
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

        # read data, potentially from cache
        if not self.reload and outputs["data"].exists():
            data_events, dy_events, bkg_events = outputs["data"].load(formatter="pickle")
        else:
            data_events, dy_events, bkg_events = self.load_data()
            outputs["data"].dump((data_events, dy_events, bkg_events), formatter="pickle")

        # initialize dictionary to store results
        # dy_weight_data[era] = {}

        norm_content = {
            0: Norm(),
            1: Norm(),
            2: Norm(),
        },

        dict_setup = {
            (2, 3): {
                "fit_str": str,
                (2, 3): norm_content,
            },
            (3, 4): {
                "fit_str": str,
                (3, 4): norm_content,
            },
            (4, 101): {
                "fit_str": str,
                (4, 5): norm_content,
                (5, 6): norm_content,
                (6, 101): norm_content,
            },
        }

        # add dummy values for both weights
        dy_events = set_ak_column(dy_events, "dy_weight_postfit", np.ones(len(dy_events), dtype=np.float32))
        dy_events = set_ak_column(dy_events, "dy_weight_norm", np.ones(len(dy_events), dtype=np.float32))

        for (njet_fit_min, njet_fit_max), fit_data in dict_setup.items():
            # filter events
            data_mask = self.get_njet_mask(data_events, njet_fit_min, njet_fit_max)
            dy_mask = self.get_njet_mask(dy_events, njet_fit_min, njet_fit_max)
            bkg_mask = self.get_njet_mask(bkg_events, njet_fit_min, njet_fit_max)

            # define variable instances
            variable_insts = [self.dilep_pt_inst, self.nbjets_inst, self.njets_inst]

            data_hists = {}
            dy_hists = {}
            bkg_hists = {}

            # get histograms for data, dy and bkg
            for var in variable_insts:
                data_hists[var.name] = self.hist_function(var, data_events[var.name][data_mask], data_events.weight[data_mask])  # noqa: E501
                dy_hists[var.name] = self.hist_function(var, dy_events[var.name][dy_mask], dy_events.weight[dy_mask])
                bkg_hists[var.name] = self.hist_function(var, bkg_events[var.name][bkg_mask], bkg_events.weight[bkg_mask])  # noqa: E501

            # TODO: ---> CREATE PDF UNWEIGHTED PLOTS

            # calculate (data-bkg)/dy ratio for dilep_pt with corresponding statistical error
            ratio_values, ratio_err, bin_centers = self.get_ratio_values(
                data_hists['dilep_pt'],
                dy_hists['dilep_pt'],
                bkg_hists['dilep_pt'],
                self.dilep_pt_inst,
            )

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

            # get post-fit parameters
            c, n, mu, sigma, a, b, r = popt

            # TODO: ---> CREATE PDF FIT PLOTS

            # build string representation and save it
            fit_str = self.get_fit_str(*popt)
            fit_data["fit_str"] = fit_str

            # evaluate and save postfit weights in new column for dy events
            dy_weight_postfit = self.get_fit_function(dy_events.gen_dilep_pt[dy_mask], *popt)
            dy_weight_postfit = ak.where(dy_mask, dy_weight_postfit, dy_events.dy_weight_postfit)
            dy_events = set_ak_column(
                dy_events,
                "dy_weight_postfit",
                dy_weight_postfit,
            )

            # get reweighted dy histogrmas
            updated_dy_weight = dy_events.weight * dy_events.dy_weight_postfit
            for var in variable_insts:
                dy_hists[var.name + "_postfit"] = self.hist_function(var, dy_events[var.name][dy_mask], updated_dy_weight)  # noqa: E501

            from IPython import embed; embed(header="debugger")

            for key, norm_data in fit_data.items():
                if not key.is_str():
                    (nbjet_min, nbjet_max) = key
                    # for btag, btag_data in norm_data.items():

                    #     mask = ...
                    #     # update hists
                    #     norm = ...

                    #     # save weights
                    #     weights2 = ak.where(mask, norm, dy_events.dy_weight_norm)
                    #     dy_events = set_ak_column(
                    #         dy_events,
                    #         "dy_weight_norm",
                    #         weights2,
                    #     )

                    # # combine both weights
                    # dy_events = set_ak_column(
                    #     dy_events,
                    #     "dy_weight",
                    #     dy_events.dy_weight_postfit * dy_events.dy_weight_norm,
                    # )

            # TODO: ---> CREATE PDF WEIGHTED PLOTS WITH : POSTFIT PT WEIGHTS

            # --------------------------------------------------------------------------------
            # 2. use nbjets to get normalization factor per njet-nbjet category and do second reweighting of DY events

            # calculate (data-bkg)/dy ratio for nbjets == 0, 1, >= 2
            # ratio_values, ratio_err, bin_centers = self.get_ratio_values(
            #     data_hists['nbjets'],
            #     dy_hists['nbjets_postfit'],
            #     bkg_hists['nbjets'],
            #     self.nbjets_inst,
            # )

            # TODO: ---> CREATE PDF WEIGHTED PLOTS WITH : POSTFIT PT WEIGHTS * NORM WEIGHTS

        # TODO: continue here :)

        # 6. Get histograms/plots with both DY weights applied: dilep_pt, nbjets_pnet_overflow, njets, etc.

        # import correctionlib.schemav2 as cs
        # from hbt.studies.dy_weights.create_clib_file import create_dy_weight_correction

    def load_data(self):
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

        return data_events, dy_events, bkg_events

    def get_njet_masks(self, events, njet_min, njet_max):
        mask = (
            (events.njets >= njet_min) &
            (events.njets < njet_max) &
            (events.channel_id == self.config_inst.channels.n.mumu.id)
        )
        return mask

    def hist_function(self, var, data, weights):
        h = create_hist_from_variables(var, weight=True)
        fill_hist(h, {var.name: data, "weight": weights})
        return h

    def get_ratio_values(
            data_h: hist.Hist,
            dy_h: hist.Hist,
            bkg_h: hist.Hist,
            variable_inst: od.Variable
    ) -> tuple[hist.Hist, hist.Hist, hist.Hist]:

        # under/overflow treatment
        for h in [data_h, dy_h, bkg_h]:
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

        # choose guassian and linear functions to do the fit
        # we cap x at 200 GeV to avoid falling off the function range
        gauss = c + (n * (1 / sigma) * np.exp(-0.5 * ((np.minimum(x,200) - mu) / sigma) ** 2))
        pol = a + b * np.minimum(x,200)

        # parameter to control the transition smoothness between the two functions
        step_par = 0.08

        # use scipy erf function to create smooth transition
        sci_erf_neg = (0.5 * (special.erf(-step_par * (np.minimum(x,200) - r)) + 1))
        sci_erf_pos = (0.5 * (special.erf(step_par * (np.minimum(x,200) - r)) + 1))

        return sci_erf_neg * gauss + sci_erf_pos * pol

    def get_fit_str(self, c, n, mu, sigma, a, b, r):
        # build fit function string
        gauss = f"(({c})+(({n})*(1/{sigma})*exp(-0.5*((min(x,200)-{mu})/{sigma})^2)))"
        pol = f"(({a})+({b})*min(x,200))"
        fit_string = f"(0.5*(erf(-0.08*(min(x,200)-{r}))+1))*{gauss}+(0.5*(erf(0.08*(min(x,200)-{r}))+1))*{pol}"

        return fit_string
