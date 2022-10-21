# coding: utf-8

"""
Producers for btag scale factor weights.
"""

from __future__ import annotations

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, safe_div
from columnflow.columnar_util import set_ak_column, flat_np_view, layout_ak_array

np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={
        "Jet.hadronFlavour", "Jet.eta", "Jet.pt", "Jet.btagDeepFlavB",
    },
    # produced columns are defined in the init function below
)
def btag_weight(
    self: Producer,
    events: ak.Array,
    jet_mask: ak.Array | type(Ellipsis) = Ellipsis,
    **kwargs,
) -> ak.Array:
    """
    B-tag scale factor weight producer. Requires an external file in the config as (e.g.)

    .. code-block:: python

        "btag_sf_corr": ("/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-d0a522ea/POG/BTV/2017_UL/btagging.json.gz", "v1"),  # noqa

    as well as an auxiliary entry in the config to refer to the b-tag correction set.

    .. code-block:: python

        cfg.x.btag_sf_correction_set = "deepJet_shape"

    In addition, JEC uncertainty sources are propagated and weight columns are written if an
    auxiliary config entry ``btag_sf_jec_sources`` exists.

    Resources:

       - https://twiki.cern.ch/twiki/bin/view/CMS/BTagShapeCalibration?rev=26
       - https://indico.cern.ch/event/1096988/contributions/4615134/attachments/2346047/4000529/Nov21_btaggingSFjsons.pdf
    """
    if self.dataset_inst.is_data:
        return events

    # get the total number of jets in the chunk
    n_jets_all = len(flat_np_view(events.Jet.pt, axis=1))

    # get flat inputs, evaluated at jet_mask
    flavor = flat_np_view(events.Jet.hadronFlavour[jet_mask], axis=1)
    abs_eta = flat_np_view(abs(events.Jet.eta[jet_mask]), axis=1)
    pt = flat_np_view(events.Jet.pt[jet_mask], axis=1)
    b_discr = flat_np_view(events.Jet.btagDeepFlavB[jet_mask], axis=1)

    # helper to create and store the weight
    def add_weight(syst_name, syst_direction, column_name):
        # define a mask that selects the correct flavor to assign to, depending on the systematic
        flavor_mask = Ellipsis
        if syst_name in ["hf", "lf", "hfstats1", "hfstats2", "lfstats1", "lfstats2"]:
            flavor_mask = flavor != 4
        elif syst_name in ["cferr1", "cferr2"]:
            flavor_mask = flavor == 4

        # get the flat scale factors
        sf_flat = self.btag_sf_corrector.evaluate(
            syst_name if syst_name == "central" else f"{syst_direction}_{syst_name}",
            flavor[flavor_mask],
            abs_eta[flavor_mask],
            pt[flavor_mask],
            b_discr[flavor_mask],
        )

        # insert them into an array of ones whose length corresponds to the total number of jets
        sf_flat_all = np.ones(n_jets_all, dtype=np.float32)
        if jet_mask is Ellipsis:
            indices = flavor_mask
        else:
            indices = flat_np_view(jet_mask)
            if flavor_mask is not Ellipsis:
                indices = np.where(indices)[0][flavor_mask]
        sf_flat_all[indices] = sf_flat

        # enforce the correct shape and create the product via all jets per event
        sf = layout_ak_array(sf_flat_all, events.Jet.pt)
        weight = ak.prod(sf, axis=1, mask_identity=False)

        # save the new column
        return set_ak_column(events, column_name, weight)

    # when the uncertainty is a known jec shift, obtain the propagated effect and do not produce
    # additional systematics
    if self.shift_inst.is_nominal:
        # nominal weight and those of all method intrinsic uncertainties
        events = add_weight("central", None, "btag_weight")
        for syst_name, col_name in self.btag_uncs.items():
            for direction in ["up", "down"]:
                name = col_name.format(year=self.config_inst.campaign.x.year)
                events = add_weight(
                    syst_name,
                    direction,
                    f"btag_weight_{name}_{direction}",
                )
    elif self.shift_is_known_jec_source:
        # TODO: year dependent jec variations covered?
        events = add_weight(
            f"jes{'' if self.jec_source == 'Total' else self.jec_source}",
            self.shift_inst.direction,
            f"btag_weight_jec_{self.jec_source}_{self.shift_inst.direction}",
        )
    else:
        # any other shift, just produce the nominal weight
        events = add_weight("central", None, "btag_weight")

    return events


@btag_weight.init
def btag_weight_init(self: Producer) -> None:
    # depending on the requested shift_inst, there are three cases to handle:
    #   1. when a JEC uncertainty is requested whose propagation to btag weights is known, the
    #      producer should only produce that specific weight column
    #   2. when the nominal shift is requested, the central weight and all variations related to the
    #      method-intrinsic shifts are produced
    #   3. when any other shift is requested, only create the central weight column
    if getattr(self, "dataset_inst", None) is None or self.dataset_inst.is_data:
        self.jec_source = None
        self.shift_is_known_jec_source = None
        self.btag_uncs = None
        return

    # to handle this efficiently in one spot, store jec information
    self.jec_source = self.shift_inst.x.jec_source if self.shift_inst.has_tag("jec") else None
    btag_sf_jec_source = "" if self.jec_source == "Total" else self.jec_source
    self.shift_is_known_jec_source = (
        self.jec_source and
        btag_sf_jec_source in self.config_inst.x("btag_sf_jec_sources", [])
    )

    # save names of method-intrinsic uncertainties
    self.btag_uncs = {
        "hf": "hf",
        "lf": "lf",
        "hfstats1": "hfstats1_{year}",
        "hfstats2": "hfstats2_{year}",
        "lfstats1": "lfstats1_{year}",
        "lfstats2": "lfstats2_{year}",
        "cferr1": "cferr1",
        "cferr2": "cferr2",
    }

    # add uncertainty sources of the method itself
    if self.shift_inst.is_nominal:
        # nominal column
        self.produces.add("btag_weight")
        # all varied columns
        for col_name in self.btag_uncs.values():
            name = col_name.format(year=self.config_inst.campaign.x.year)
            for direction in ["up", "down"]:
                self.produces.add(f"btag_weight_{name}_{direction}")
    elif self.shift_is_known_jec_source:
        # jec varied column
        self.produces.add(f"btag_weight_jec_{self.jec_source}_{self.shift_inst.direction}")
    else:
        # only the nominal column
        self.produces.add("btag_weight")


@btag_weight.requires
def btag_weight_requires(self: Producer, reqs: dict) -> None:
    if self.dataset_inst.is_data or "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@btag_weight.setup
def btag_weight_setup(self: Producer, reqs: dict, inputs: dict) -> None:
    if self.dataset_inst.is_data:
        self.btag_sf_corrector = None
        return

    bundle = reqs["external_files"]

    # create the btag sf corrector
    import correctionlib
    correction_set = correctionlib.CorrectionSet.from_string(
        bundle.files.btag_sf_corr.load(formatter="gzip").decode("utf-8"),
    )
    self.btag_sf_corrector = correction_set[self.config_inst.x.btag_sf_correction_set]


@producer(
    uses={
        btag_weight.PRODUCES, "process_id", "Jet.pt",
    },
    # produced columns are defined in the init function below
)
def normalized_btag_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    if self.dataset_inst.is_data:
        return events

    for weight_name in self[btag_weight].produces:
        if not weight_name.startswith("btag_weight"):
            continue

        # create a weight vectors starting with ones for both weight variations, i.e.,
        # nomalization per pid and normalization per pid and jet multiplicity
        norm_weight_per_pid = np.ones(len(events), dtype=np.float32)
        norm_weight_per_pid_njet = np.ones(len(events), dtype=np.float32)

        # fill weights with a new mask per unique process id (mostly just one)
        for pid in self.unique_process_ids:
            pid_mask = events.process_id == pid
            # single value
            norm_weight_per_pid[pid_mask] = self.ratio_per_pid[weight_name][pid]
            # lookup table
            n_jets = ak.num(events[pid_mask].Jet.pt, axis=1)
            norm_weight_per_pid_njet[pid_mask] = self.ratio_per_pid_njet[weight_name][pid][n_jets]

        # multiply with actual weight
        norm_weight_per_pid = norm_weight_per_pid * events[weight_name]
        norm_weight_per_pid_njet = norm_weight_per_pid_njet * events[weight_name]

        # store them
        events = set_ak_column(events, f"normalized_{weight_name}", norm_weight_per_pid)
        events = set_ak_column(events, f"normalized_njet_{weight_name}", norm_weight_per_pid_njet)

    return events


@normalized_btag_weight.init
def normalized_btag_weight_init(self: Producer) -> None:
    if getattr(self, "dataset_inst", None) is None or self.dataset_inst.is_data:
        return

    # for btag weight, declare that one normalized version of the weight is produced
    for weight_name in self[btag_weight].produces:
        if not weight_name.startswith("btag_weight"):
            continue

        self.produces.add(f"normalized_{weight_name}")
        self.produces.add(f"normalized_njet_{weight_name}")


@normalized_btag_weight.requires
def normalized_btag_weight_requires(self: Producer, reqs: dict) -> None:
    # do nothing for data
    if self.dataset_inst.is_data:
        return

    from columnflow.tasks.selection import MergeSelectionStats
    reqs["selection_stats"] = MergeSelectionStats.req(
        self.task,
        tree_index=0,
        branch=-1,
        _exclude=MergeSelectionStats.exclude_params_forest_merge,
    )


@normalized_btag_weight.setup
def normalized_btag_weight_setup(self: Producer, reqs: dict, inputs: dict) -> None:
    # set to None for data
    if self.dataset_inst.is_data:
        self.unique_process_ids = None
        self.ratio_per_pid = None
        self.ratio_per_pid_njet = None
        return

    # load the selection stats
    stats = inputs["selection_stats"]["collection"][0].load(formatter="json")

    # get the unique process ids in that dataset
    key = "sum_mc_weight_selected_no_bjet_per_process_and_njet"
    self.unique_process_ids = list(map(int, stats[key].keys()))

    # get the maximum numbers of jets
    max_n_jets = max(map(int, sum((list(d.keys()) for d in stats[key].values()), [])))

    # helper to get numerators and denominators
    def numerator_per_pid(pid):
        key = "sum_mc_weight_selected_no_bjet_per_process"
        return stats[key].get(str(pid), 0.0)

    def denominator_per_pid(weight_name, pid):
        key = f"sum_mc_weight_{weight_name}_selected_no_bjet_per_process"
        return stats[key].get(str(pid), 0.0)

    def numerator_per_pid_njet(pid, n_jets):
        key = "sum_mc_weight_selected_no_bjet_per_process_and_njet"
        d = stats[key].get(str(pid), {})
        return d.get(str(n_jets), 0.0)

    def denominator_per_pid_njet(weight_name, pid, n_jets):
        key = f"sum_mc_weight_{weight_name}_selected_no_bjet_per_process_and_njet"
        d = stats[key].get(str(pid), {})
        return d.get(str(n_jets), 0.0)

    # extract the ratio per weight and pid
    self.ratio_per_pid = {
        weight_name: {
            pid: safe_div(numerator_per_pid(pid), denominator_per_pid(weight_name, pid))
            for pid in self.unique_process_ids
        }
        for weight_name in self[btag_weight].produces
        if weight_name.startswith("btag_weight")
    }

    # extract the ratio per weight, pid and also the jet multiplicity, using the latter as in index
    # for a lookup table (since it naturally starts at 0)
    self.ratio_per_pid_njet = {
        weight_name: {
            pid: np.array([
                safe_div(numerator_per_pid_njet(pid, n_jets), denominator_per_pid_njet(weight_name, pid, n_jets))
                for n_jets in range(max_n_jets + 1)
            ])
            for pid in self.unique_process_ids
        }
        for weight_name in self[btag_weight].produces
        if weight_name.startswith("btag_weight")
    }
