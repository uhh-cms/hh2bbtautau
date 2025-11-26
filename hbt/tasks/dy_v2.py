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
import correctionlib

from columnflow.tasks.framework.base import TaskShifts, ConfigTask
from columnflow.tasks.framework.mixins import (
    DatasetsProcessesMixin, ProducerClassesMixin, CalibratorClassesMixin,
    SelectorClassMixin, ReducerClassMixin,
)
from columnflow.tasks.production import ProduceColumns
from columnflow.plotting.plot_functions_1d import plot_variable_stack
from columnflow.tasks.reduction import ProvideReducedEvents
from columnflow.util import maybe_import
from columnflow.columnar_util import (
    ChunkedIOHandler, RouteFilter, update_ak_array, attach_coffea_behavior, layout_ak_array,
    set_ak_column,
)
from columnflow.hist_util import create_hist_from_variables, fill_hist

import correctionlib.schemav2 as cs

from hbt.tasks.base import HBTTask

import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
from typing import Literal

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
    DatasetsProcessesMixin
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
            (self.dilep_pt_inst, 'dilep_pt'),
            (self.nbjets_inst, 'nbjets')
            # (self.njets_inst, 'njets')
        ]
        self.variables_names = [var_name for _, var_name in self.variables]

        self.channels = ["mumu"]  # ["mumu", "ee"]
        self.cats = ["eq2j", "eq3j", "eq4j", "eq5j", "ge4j", "ge6j"]
        self.postfixes = [
            "", "_postfit", "_gauss_up", "_gauss_down", "_lin_up", "_lin_down",
            "_norm_postfit", "_norm_gauss_up", "_norm_gauss_down", "_norm_lin_up", "_norm_lin_down"
        ]

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

    def output(self):
        outputs = {
            "weights": self.target("weights.pkl"),
            "plots": {},
        }

        # define plot outputs for fit functions
        for cat in ["eq2j", "eq3j", "ge4j"]:
            identifier = f"fit_{cat}"
            outputs["plots"][identifier] = self.target(f"{identifier}.pdf")

        # define plot outputs for kinematic variables
        for channel in self.channels:
            for var_name in self.variables_names:
                for cat in self.cats:
                    for postfix in self.postfixes:
                        identifier = f"{channel}_{var_name}{postfix}_{cat}"
                        outputs["plots"][identifier] = self.target(f"{identifier}.pdf")

        return outputs

    def requires(self):
        return LoadDYData.req(self)

    def run(self):
        outputs = self.output()

        # read data, potentially from cache
        data_events, dy_events, bkg_events = self.input().load(formatter="pickle")

        # initialize dictionary to store results
        dy_weight_data = {}

        norm_content: dict[int, Norm | None] = {
            (0, 1): None,
            (1, 2): None,
            (2, 101): None,
        }

        dict_setup = {
            (2, 3): {
                "fit_str": str,
                "gauss_up": {"fit": str, (2, 3): {}},
                "gauss_down": {"fit": str, (2, 3): {}},
                "lin_up": {"fit": str, (2, 3): {}},
                "lin_down": {"fit": str, (2, 3): {}},
                (2, 3): {},
            },
            (3, 4): {
                "fit_str": str,
                "gauss_up": {"fit": str, (3, 4): {}},
                "gauss_down": {"fit": str, (3, 4): {}},
                "lin_up": {"fit": str, (3, 4): {}},
                "lin_down": {"fit": str, (3, 4): {}},
                (3, 4): {},
            },
            (4, 101): {
                "fit_str": str,
                "gauss_up": {
                    "fit": str,
                    (4, 5): {},
                    (5, 6): {},
                    (6, 101): {}
                },
                "gauss_down": {
                    "fit": str,
                    (4, 5): {},
                    (5, 6): {},
                    (6, 101): {}
                },
                "lin_up": {
                    "fit": str,
                    (4, 5): {},
                    (5, 6): {},
                    (6, 101): {}
                },
                "lin_down": {
                    "fit": str,
                    (4, 5): {},
                    (5, 6): {},
                    (6, 101): {}
                },
                (4, 5): {},
                (5, 6): {},
                (6, 101): {},
            },
        }

        # add dummy values for weight columns to be updated later
        dy_events = set_ak_column(dy_events, "dy_weight_postfit", np.ones(len(dy_events), dtype=np.float32))
        dy_events = set_ak_column(dy_events, "dy_weight_norm", np.ones(len(dy_events), dtype=np.float32))
        dy_events = set_ak_column(dy_events, "dy_weight_gauss_up", np.ones(len(dy_events), dtype=np.float32))
        dy_events = set_ak_column(dy_events, "dy_weight_gauss_down", np.ones(len(dy_events), dtype=np.float32))
        dy_events = set_ak_column(dy_events, "dy_weight_lin_up", np.ones(len(dy_events), dtype=np.float32))
        dy_events = set_ak_column(dy_events, "dy_weight_lin_down", np.ones(len(dy_events), dtype=np.float32))

        ###############################################################################
        # fill dict_setup with fit functions and btag Norm content
        for (njet_fit_min, njet_fit_max), fit_data in dict_setup.items():

            # categories for dilep_pt fit
            cat_fit = f"eq{njet_fit_min}j" if njet_fit_min < 4 else "ge4j"

            data_hists = {}
            dy_hists = {}
            bkg_hists = {}

            # get pre-fit hists and plots
            original_dy_weights = dy_events.weight
            for key, norm_data in fit_data.items():
                if isinstance(key, str):
                    continue
                (njet_min, njet_max) = key
                cat_bin = f"eq{njet_min}j" if njet_min < 6 else "ge6j"
                [(data_hists, data_events), (dy_hists, dy_events), (bkg_hists, bkg_events)] = self.get_hists_and_plots(  # noqa: E501
                    data_hists, dy_hists, bkg_hists,
                    data_events, dy_events, bkg_events,
                    njet_fit_min, njet_fit_max, fit_data, original_dy_weights,
                    cat_bin,
                    self.variables,
                    "",
                )

            # get pre-fit hists and plots for ge4j category
            if cat_fit == "ge4j":
                [(data_hists, data_events), (dy_hists, dy_events), (bkg_hists, bkg_events)] = self.get_hists_and_plots(
                    data_hists, dy_hists, bkg_hists,
                    data_events, dy_events, bkg_events,
                    njet_fit_min, njet_fit_max, fit_data, original_dy_weights,
                    cat_fit,
                    self.variables,
                    "",
                )

            ##############################################################
            # calculate (data-bkg)/dy ratio in dilep_pt using mumu channel
            ratio_values, ratio_err, bin_centers = self.get_ratio_values(
                data_hists[f'mumu_dilep_pt_{cat_fit}'],
                dy_hists[f'mumu_dilep_pt_{cat_fit}'],
                bkg_hists[f'mumu_dilep_pt_{cat_fit}'],
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

            # get post-fit parameters and build string of the fit function
            c, n, mu, sigma, a, b, r = popt
            fit_str = self.get_fit_str(*popt)
            fit_data["fit_str"] = fit_str

            fit_params = {
                "postfit": popt,
            }

            # -------------------------------------------------------------------------------------------------
            # get bin mask: True for gaussian like bins, False for linear like bins
            bin_mask = bin_centers <= r

            # define shifted ratios by 95% CL
            ratio_values_gauss_up = np.where(bin_mask, ratio_values + 2 * ratio_err, ratio_values)
            ratio_values_gauss_down = np.where(bin_mask, ratio_values - 2 * ratio_err, ratio_values)
            ratio_values_lin_up = np.where(~bin_mask, ratio_values + 2 * ratio_err, ratio_values)
            ratio_values_lin_down = np.where(~bin_mask, ratio_values - 2 * ratio_err, ratio_values)
            shifted_ratios = [
                (ratio_values_gauss_up, "gauss_up"),
                (ratio_values_gauss_down, "gauss_down"),
                (ratio_values_lin_up, "lin_up"),
                (ratio_values_lin_down, "lin_down")
            ]

            # redo the fit for each shifted scenario and save to dict
            for ratio_values_shifted, fit_name in shifted_ratios:
                popt, pcov = optimize.curve_fit(
                    self.get_fit_function,
                    bin_centers,
                    ratio_values_shifted,
                    p0=starting_values, method="trf",
                    sigma=np.maximum(ratio_err, 1e-5),
                    absolute_sigma=True,
                    bounds=(lower_bounds, upper_bounds),
                )

                # save shifted fit string to dict
                fit_str_shifted = self.get_fit_str(*popt)
                fit_data[f"{fit_name}"]["fit"] = fit_str_shifted

                # save new fit parameters for later use
                fit_params[f"{fit_name}"] = popt
            # -------------------------------------------------------------------------------------------------

            # plot the fit result including shifted scenarios
            self.get_fit_plot(cat_fit, self.get_fit_function, fit_params, ratio_values, ratio_err, bin_centers)

            # get postfit weights using fit function with nominal and shifted parameters
            dy_njet_mask = self.get_njet_mask(dy_events, njet_fit_min, njet_fit_max)
            dy_weight_postfit = ak.where(
                dy_njet_mask,
                self.get_fit_function(dy_events.gen_dilepton_pt, *fit_params["postfit"]),
                dy_events.dy_weight_postfit,
            )
            dy_weight_gauss_up = ak.where(
                dy_njet_mask,
                self.get_fit_function(dy_events.gen_dilepton_pt, *fit_params["gauss_up"]),
                dy_events.dy_weight_gauss_up,
            )
            dy_weight_gauss_down = ak.where(
                dy_njet_mask,
                self.get_fit_function(dy_events.gen_dilepton_pt, *fit_params["gauss_down"]),
                dy_events.dy_weight_gauss_down,
            )
            dy_weight_lin_up = ak.where(
                dy_njet_mask,
                self.get_fit_function(dy_events.gen_dilepton_pt, *fit_params["lin_up"]),
                dy_events.dy_weight_lin_up,
            )
            dy_weight_lin_down = ak.where(
                dy_njet_mask,
                self.get_fit_function(dy_events.gen_dilepton_pt, *fit_params["lin_down"]),
                dy_events.dy_weight_lin_down,
            )

            new_columns = {
                "postfit": dy_weight_postfit,
                "gauss_up": dy_weight_gauss_up,
                "gauss_down": dy_weight_gauss_down,
                "lin_up": dy_weight_lin_up,
                "lin_down": dy_weight_lin_down,
            }

            updated_weights = {}

            # update event info with new columns, get hists and plots
            for shift_name, column_obj in new_columns.items():
                dy_events = set_ak_column(
                    dy_events,
                    f"dy_weight_{shift_name}",
                    column_obj,
                )

                tmp_dy_weights = dy_events["weight"] * dy_events[f"dy_weight_{shift_name}"]
                updated_weights[f"{shift_name}"] = tmp_dy_weights

                [(data_hists, data_events), (dy_hists, dy_events), (bkg_hists, bkg_events)] = self.get_hists_and_plots(
                    data_hists, dy_hists, bkg_hists,
                    data_events, dy_events, bkg_events,
                    njet_fit_min, njet_fit_max, fit_data, updated_weights[f"{shift_name}"],
                    cat_fit,
                    self.variables,
                    f"_{shift_name}",
                )

            ##############################################################
            # calculate btag normalizations

            for key, norm_data in fit_data.items():

                # calculate btag normalizations for shifted fit scenarios
                if isinstance(key, str):
                    if key == "fit_str":
                        continue
                    else:
                        for tmp_key, tmp_norm_data in norm_data.items():
                            if isinstance(tmp_key, str):
                                continue
                            (njet_min, njet_max) = tmp_key
                            cat_bin = f"eq{njet_min}j" if njet_min < 6 else "ge6j"
                            if cat_bin in ["eq4j", "eq5j", "ge6j"]:
                                [(data_hists, data_events), (dy_hists, dy_events), (bkg_hists, bkg_events)] = self.get_hists_and_plots(  # noqa: E501
                                    data_hists, dy_hists, bkg_hists,
                                    data_events, dy_events, bkg_events,
                                    njet_fit_min, njet_fit_max, tmp_norm_data, updated_weights[f"{key}"],
                                    cat_bin,
                                    self.variables,
                                    f"_{key}",
                                )
                            identifier = f"mumu_nbjets_{key}_{cat_bin}"
                            ratio_values, _, _ = self.get_ratio_values(
                                data_hists[identifier],
                                dy_hists[identifier],
                                bkg_hists[identifier],
                                self.nbjets_inst,
                            )
                            tmp_norm_data = norm_content.copy()
                            for btag, btag_data in tmp_norm_data.items():
                                n_btag = btag[0]

                                # save normalizations with statistical uncertainty
                                tmp_norm_data[btag] = Norm(ratio_values[n_btag], 0.0)  # no stat uncert for shifted fits
                                btag_mask = (
                                    (self.get_njet_mask(dy_events, njet_min, njet_max)) &
                                    (dy_events.nbjets == n_btag if n_btag < 2 else dy_events.nbjets >= n_btag)
                                )

                                dy_weight_norm_shifted = ak.where(
                                    btag_mask,
                                    tmp_norm_data[btag].nom,  # use nominal shift for now
                                    dy_events.dy_weight_norm)  # initial weight of 1.0

                                # set new column with btag normalization for shifted fit
                                dy_events = set_ak_column(
                                    dy_events,
                                    f"dy_weight_norm_{key}",
                                    dy_weight_norm_shifted,
                                )

                                # save DY event weights for shifted fit scenario
                                norm_dy_weights = dy_events["weight"] * dy_events[f"dy_weight_{key}"] * dy_events[f"dy_weight_norm_{key}"]  # noqa: E501
                                updated_weights[f"norm_{key}"] = norm_dy_weights

                                fit_data[key][tmp_key] = tmp_norm_data
                        continue
                # -----------------------------------------------------------------------------
                # calculate btag normalizations for nominal fit scenario
                (njet_min, njet_max) = key
                cat_bin = f"eq{njet_min}j" if njet_min < 6 else "ge6j"

                # get postfit nbjets hists and plots for missing categories (eq4j, eq5j, ge6j)
                if cat_bin in ["eq4j", "eq5j", "ge6j"]:
                    [(data_hists, data_events), (dy_hists, dy_events), (bkg_hists, bkg_events)] = self.get_hists_and_plots(  # noqa: E501
                        data_hists, dy_hists, bkg_hists,
                        data_events, dy_events, bkg_events,
                        njet_fit_min, njet_fit_max, fit_data, updated_weights["postfit"],
                        cat_bin,
                        self.variables,
                        "_postfit",
                    )

                identifier = f"mumu_nbjets_postfit_{cat_bin}"
                ratio_values, ratio_err, bin_centers = self.get_ratio_values(
                    data_hists[identifier],
                    dy_hists[identifier],
                    bkg_hists[identifier],
                    self.nbjets_inst,
                )

                # fill norm_data per btag bin
                norm_data = norm_content.copy()
                for btag, btag_data in norm_data.items():
                    n_btag = btag[0]

                    # save normalizations with statistical uncertainty
                    norm_data[btag] = Norm(ratio_values[n_btag], ratio_err[n_btag])
                    btag_mask = (
                        (self.get_njet_mask(dy_events, njet_min, njet_max)) &
                        (dy_events.nbjets == n_btag if n_btag < 2 else dy_events.nbjets >= n_btag)
                    )
                    dy_weight_norm = ak.where(
                        btag_mask,
                        norm_data[btag].nom,  # use nominal shift for now
                        dy_events.dy_weight_norm)
                    dy_events = set_ak_column(
                        dy_events,
                        "dy_weight_norm",
                        dy_weight_norm,
                    )

                # save normalizations in corresponding njet bin
                fit_data[key] = norm_data
                # -----------------------------------------------------------------------------

            # save DY event weights for nominal fit scenario
            updated_weights["norm_postfit"] = dy_events.weight * dy_events.dy_weight_postfit * dy_events.dy_weight_norm

            # get norm hists and plots for nominal and shifted fit scenarios
            for key, norm_data in fit_data.items():
                if isinstance(key, str):
                    if key == "fit_str":
                        continue
                    else:
                        for tmp_key, tmp_norm_data in norm_data.items():
                            if isinstance(tmp_key, str):
                                continue
                            (njet_min, njet_max) = tmp_key
                            cat_bin = f"eq{njet_min}j" if njet_min < 6 else "ge6j"
                            [(data_hists, data_events), (dy_hists, dy_events), (bkg_hists, bkg_events)] = self.get_hists_and_plots(  # noqa: E501
                                data_hists, dy_hists, bkg_hists,
                                data_events, dy_events, bkg_events,
                                njet_fit_min, njet_fit_max, fit_data, updated_weights[f"norm_{key}"],
                                cat_bin,
                                self.variables,
                                f"_norm_{key}",
                            )
                        continue
                (njet_min, njet_max) = key
                cat_bin = f"eq{njet_min}j" if njet_min < 6 else "ge6j"
                [(data_hists, data_events), (dy_hists, dy_events), (bkg_hists, bkg_events)] = self.get_hists_and_plots(  # noqa: E501
                    data_hists, dy_hists, bkg_hists,
                    data_events, dy_events, bkg_events,
                    njet_fit_min, njet_fit_max, fit_data, updated_weights["norm_postfit"],
                    cat_bin,
                    self.variables,
                    "_norm_postfit",
                )

            # get norm hists and plots for ge4j category for nominal and shifted fit scenarios
            if cat_fit == "ge4j":
                for shift_name, _ in updated_weights.items():
                    if shift_name.startswith("norm"):
                        [(data_hists, data_events), (dy_hists, dy_events), (bkg_hists, bkg_events)] = self.get_hists_and_plots(  # noqa: E501
                            data_hists, dy_hists, bkg_hists,
                            data_events, dy_events, bkg_events,
                            njet_fit_min, njet_fit_max, fit_data, updated_weights[f"{shift_name}"],
                            cat_fit,
                            self.variables,
                            f"_{shift_name}",
                        )

        ###############################################################################
        # restructure dict_setup for correction lib json, including btag up/down shifts
        dy_weight_data = self.get_dy_weight_data(dict_setup)

        # save final dy weights
        outputs["weights"].dump(dy_weight_data, formatter="pickle")

    def get_njet_mask(self, events, njet_min, njet_max, channel=None):
        mask = ((events.njets >= njet_min) & (events.njets < njet_max))
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
            variable_inst: od.Variable
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

    def get_fit_plot(self, cat, fit_function, fit_params, ratio_values, ratio_err, bin_centers):
        outputs = self.output()

        # create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Nominal fit
        s = np.linspace(0, 200, 1000)
        y_nom = [fit_function(v, *fit_params["postfit"]) for v in s]
        ax.plot(s, y_nom, color="black", label="Nominal", lw=2)

        # shifted fits
        for regime, colour, label in [("gauss", "red", "Gaussian (up/down)"), ("lin", "blue", "Linear (up/down)")]:
            y = [self.get_fit_function(v, *fit_params[f"{regime}_up"]) for v in s]
            ax.fill_between(s, y, y_nom, color=colour, alpha=0.2, label=label)
            # ax.plot(s, y, color=colour, label=label, lw=1, linestyle="--")
            y = [self.get_fit_function(v, *fit_params[f"{regime}_down"]) for v in s]
            ax.fill_between(s, y, y_nom, color=colour, alpha=0.2)

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
        # get era, luminosity /pb -> /fb and com
        lumi = "8.0"
        com = "13.6 TeV"

        # build title
        fig.subplots_adjust(top=0.93)
        if cat != "ge4j":
            njets = int(cat.replace("eq", "").replace("j", ""))
            jet_label = rf"$\mu\mu$ channel, DY region, AK4 jets $={njets}$"
        else:
            jet_label = r"$\mu\mu$ channel, DY region, AK4 jets  $\geq 4$"
        label = rf"Private work (CMS data/simulation)      {jet_label}"
        # styling
        ax.legend(loc="lower right")
        ax.set_xlabel(r"$\mathrm{p}_{\mathrm{T,ll}} \ [\mathrm{GeV}]$", fontsize=15, loc="right")
        ax.set_ylabel("Data - MC / DY", fontsize=15, loc="top")
        ax.grid(True)
        fig.text(0.12, 0.97, label, verticalalignment='top', horizontalalignment='left', fontsize=13)
        fig.text(0.90, 0.97, f"{lumi} fb$^{{-1}}$ ({com})", verticalalignment='top', horizontalalignment='right', fontsize=13)
        ax.tick_params(axis='both', labelsize=15)

        # save plot
        identifier = f"fit_{cat}"
        outputs["plots"][identifier].dump(fig, formatter="mpl")

    def get_dy_weight_data(self, dict_setup: dict):
        """
        Function to build the nested dictionary structure to export DY weight info to a correction lib json.
        """
        dict_data = {}

        # fill nominal dict first with nominal btag normalizations and fit functions
        dict_data["nom"] = {}
        # loop over njet bins
        for njet_bin in dict_setup.keys():
            fit_string = dict_setup[njet_bin]["fit_str"]
            nbjet_bins = list(dict_setup[njet_bin])[1:]
            # loop over btag bins
            for nbjet_bin in nbjet_bins:
                dict_data["nom"][nbjet_bin] = {}
                for btag in dict_setup[njet_bin][nbjet_bin].keys():
                    norm_var = dict_setup[njet_bin][nbjet_bin][btag]
                    norm_value = getattr(norm_var, "nom")
                    expr = [(0.0, float("inf"), f"{norm_value}*({fit_string})")]
                    # save expression in nested dict
                    dict_data["nom"][nbjet_bin][btag] = expr

        # fill statistical up/down btag shifts considering # btag as separate sources of uncertainty
        nbjet_bins = dict_data["nom"].keys()
        btag_bins = dict_data["nom"][nbjet_bin].keys()
        for btag_bin in btag_bins:
            btag = btag_bin[0]  # get btag value: 0, 1, 2
            for direction in ["up", "down"]:
                # create new uncertainty entry in dict_data
                shift_str = f"stat_btag{btag}_{direction}"
                dict_data[shift_str] = {}
                # shift 0/1/2 btag entry for all njet entries at once
                for njet_bin in dict_setup.keys():
                    fit_string = dict_setup[njet_bin]["fit_str"]
                    updated_content = {}
                    for nbjet_bin in nbjet_bins:
                        # only update if corresponding nbjet_bin exists
                        if nbjet_bin not in dict_setup[njet_bin]:
                            continue
                        dict_data[shift_str][nbjet_bin] = {}

                        # get shifted btag normalization with corresponfing fit function string
                        norm_var = dict_setup[njet_bin][nbjet_bin][btag_bin]
                        norm_value = getattr(norm_var, direction)
                        fit_string = dict_setup[njet_bin]["fit_str"]
                        expr = [(0.0, float("inf"), f"{norm_value}*({fit_string})")]

                        # copy nominal content, updating only the corresponding btag entry
                        updated_content[btag_bin] = expr
                        shifted_dict = law.util.merge_dicts(
                            dict_data["nom"][nbjet_bin],
                            updated_content, deep=True
                        )

                        # save shifted dict
                        dict_data[shift_str][nbjet_bin] = shifted_dict

        # fill up/down fit shifts considering gaussian and linear functions as separate sources of uncertainty
        for regime in ["gauss", "lin"]:
            for direction in ["up", "down"]:
                shift_str = f"{regime}_{direction}"
                dict_data[shift_str] = {}

                # get corresponding shifted fit string per njet_bin
                for njet_bin in dict_setup.keys():
                    fit_string = dict_setup[njet_bin][f"{shift_str}"]["fit"]

                    # get corresponding nominal btag normalizations per nbjet_bin
                    for nbjet_bin in nbjet_bins:
                        if nbjet_bin not in dict_setup[njet_bin][f"{shift_str}"]:
                            continue
                        dict_data[shift_str][nbjet_bin] = {}

                        # loop over 0,1 and 2 btag bins
                        for btag_bin in btag_bins:
                            norm_var = dict_setup[njet_bin][f"{shift_str}"][nbjet_bin][btag_bin]
                            norm_value = getattr(norm_var, "nom")
                            expr = [(0.0, float("inf"), f"{norm_value}*({fit_string})")]
                            dict_data[shift_str][nbjet_bin][btag_bin] = expr

        return dict_data

    def get_hists_and_plots(
        self,
        data_hists,
        dy_hists,
        bkg_hists,
        data_events,
        dy_events,
        bkg_events,
        njet_fit_min,
        njet_fit_max,
        fit_data,
        dy_weights,
        cat,
        variables: list[tuple[od.Variable, str]],
        postfix: Literal[
            "", "_postfit", "_gauss_up", "_gauss_down", "_lin_up", "_lin_down",
            "_norm_postfit", "_norm_gauss_up", "_norm_gauss_down", "_norm_lin_up", "_norm_lin_down"
        ],
    ):
        outputs = self.output()
        bkg_process = od.Process(name="Backgrounds", id="+", color1="#e76300")

        for channel in self.channels:
            # update hists
            for hist, events in [(data_hists, data_events), (dy_hists, dy_events), (bkg_hists, bkg_events)]:
                # get hists for categories (eq2j, eq3j, eq4j, eq5j, ge6j)
                if cat != "ge4j":
                    for key, norm_data in fit_data.items():
                        if isinstance(key, str):
                            continue
                        (njet_min, njet_max) = key
                        cat_bin = f"eq{njet_min}j" if njet_min < 6 else "ge6j"
                        event_mask = self.get_njet_mask(events, njet_min, njet_max, channel)
                        cat_label = f"{channel}__dyc__{cat_bin}__ge0b__os"
                        for (var, var_name) in variables:
                            identifier = f"{channel}_{var_name}{postfix}_{cat_bin}"
                            if hist is dy_hists:
                                hist[identifier] = self.hist_function(var, events[var_name][event_mask], dy_weights[event_mask])  # noqa: E501
                            else:
                                hist[identifier] = self.hist_function(var, events[var_name][event_mask], events.weight[event_mask])  # noqa: E501
                elif cat == "ge4j":
                    event_mask = self.get_njet_mask(events, njet_fit_min, njet_fit_max, channel)
                    cat_label = f"{channel}__dyc__{cat}__ge0b__os"
                    for (var, var_name) in variables:
                        identifier = f"{channel}_{var_name}{postfix}_{cat}"
                        if hist is dy_hists:
                            hist[identifier] = self.hist_function(var, events[var_name][event_mask], dy_weights[event_mask])  # noqa: E501
                        else:
                            hist[identifier] = self.hist_function(var, events[var_name][event_mask], events.weight[event_mask])  # noqa: E501

            # plotting each variable
            for (var, var_name) in variables:
                identifier = f"{channel}_{var_name}{postfix}_{cat}"
                hists_to_plot = {
                    self.config_inst.get_process("dy"): dy_hists[identifier],
                    self.config_inst.get_process("data"): data_hists[identifier],
                    bkg_process: bkg_hists[identifier],
                }
                fig, _ = plot_variable_stack(
                    hists=hists_to_plot,
                    config_inst=self.config_inst,
                    category_inst=self.config_inst.get_category(cat_label),
                    variable_insts=[var,],
                    shift_insts=[self.config_inst.get_shift("nominal")],
                )
                outputs["plots"][identifier].dump(fig, formatter="mpl")

        updated_data = [(data_hists, data_events), (dy_hists, dy_events), (bkg_hists, bkg_events)]

        return updated_data


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
            description="DY weights derived in the phase space of the hh2bbtautau analysis, supposed to correct njet and "
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


class EvaluateDYWeights(DYBaseTask):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.variables = [
            (self.nbjets_inst, "nbjets")
        ]

        self.era = self.config_inst.x.dy_weight_config.era
        self.channels = ["mumu"]  # , "ee"]
        self.cat_def = [("ge2j", 2, 101)]  # , ("eq2j", 2, 3), ("eq3j", 3, 4), ("ge4j", 4, 101)]

        # load DY weight corrections from json file
        self.dy_file = "/data/dust/user/alvesand/analysis/hh2bbtautau_data/hbt_store/analysis_hbt/hbt.ExportDYWeights/22pre_v14/prod20_vbf_dy_fix_v2/hbt_corrections.json.gz"  # noqa: E501
        self.dy_correction = correctionlib.CorrectionSet.from_file(self.dy_file)
        self.correction_set = self.dy_correction["dy_weight"]

    def requires(self):
        return LoadDYData.req(self)

    def output(self):
        outputs = {}

        identifier = [
            f"{channel}__dyc__{cat}__ge0b__os__{var_name}"
            for channel in self.channels
            for cat, njet_min, njet_max in self.cat_def
            for _, var_name in self.variables
        ]

        for id_str in identifier:
            outputs[id_str] = self.target(f"{id_str}.pdf")

        return outputs

    def hist_function(self, var, data, weights):
        h = create_hist_from_variables(var, weight=True)
        fill_hist(h, {var.name: data, "weight": weights})
        return h

    def run(self):
        outputs = self.output()

        # read data, potentially from cache
        data_events, dy_events, bkg_events = self.input().load(formatter="pickle")

        data_hists = {}
        dy_hists = {}
        bkg_hists = {}
        bkg_process = od.Process(name="Backgrounds", id="+", color1="#e76300")

        # get DY updated weights
        syst = "nom"
        dy_n_jet = dy_events.njets
        dy_n_tag = dy_events.nbjets
        dy_ll_pt = dy_events.gen_dilepton_pt
        dy_sf = self.correction_set.evaluate(self.era, dy_n_jet, dy_n_tag, dy_ll_pt, syst)
        clib_weight = dy_events.weight * dy_sf

        print("\n--> Done evaluating DY weights using following file:")
        print(self.dy_file.replace("/data/dust/user/alvesand/analysis/hh2bbtautau_data/hbt_store/analysis_hbt/", ""))

        channel = "mumu"
        channel_id = self.config_inst.channels.n.mumu.id

        for cat, njet_min, njet_max in self.cat_def:
            cat_label = f"{channel}__dyc__{cat}__ge0b__os"

            # get hists
            for hist, events in [(data_hists, data_events), (dy_hists, dy_events), (bkg_hists, bkg_events)]:
                for (var, var_name) in self.variables:
                    identifier = f"{channel}__dyc__{cat}__ge0b__os__{var_name}"
                    event_mask = (events.njets >= njet_min) & (events.njets < njet_max) & (events.channel_id == channel_id)  # noqa: E501
                    if hist is dy_hists:
                        hist[identifier] = self.hist_function(var, events[var_name][event_mask], clib_weight[event_mask])  # noqa: E501
                    else:
                        hist[identifier] = self.hist_function(var, events[var_name][event_mask], events.weight[event_mask])  # noqa: E501

            # get plots
            for (var, var_name) in self.variables:
                identifier = f"{channel}__dyc__{cat}__ge0b__os__{var_name}"

                hists_to_plot = {
                    self.config_inst.get_process("dy"): dy_hists[identifier],
                    self.config_inst.get_process("data"): data_hists[identifier],
                    bkg_process: bkg_hists[identifier],
                }

                fig, _ = plot_variable_stack(
                    hists=hists_to_plot,
                    config_inst=self.config_inst,
                    category_inst=self.config_inst.get_category(cat_label),
                    variable_insts=[var,],
                    shift_insts=[self.config_inst.get_shift("nominal")],
                )
                outputs[identifier].dump(fig, formatter="mpl")
