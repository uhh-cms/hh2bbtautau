# coding: utf-8

"""
Custom trigger scale factor production.
"""

import functools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, InsertableDict
from columnflow.columnar_util import set_ak_column, flat_np_view, layout_ak_array
from colmnflow.production.cms.muon import muon_weights
from columnflow.production.cms.electron import electron_weights


ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


# subclass the electron weights producer to create the electron efficiencies
single_electron_trigger_data_effs = electron_weights.derive(
    "single_electron_trigger_data_effs",
    cls_dict={
        "get_electron_file": (lambda self, external_files: external_files.electron_trigger_sf),
        "get_electron_config": (lambda self: self.config_inst.x.single_electron_trigger_data_effs_names),
        "weight_name": "single_electron_trigger_data_effs",
    },
)

single_electron_trigger_mc_effs = electron_weights.derive(
    "single_electron_trigger_mc_effs",
    cls_dict={
        "get_electron_file": (lambda self, external_files: external_files.electron_trigger_sf),
        "get_electron_config": (lambda self: self.config_inst.x.single_electron_trigger_mc_effs_names),
        "weight_name": "single_electron_trigger_mc_effs",
    },
)

cross_electron_trigger_data_effs = electron_weights.derive(
    "cross_electron_trigger_data_effs",
    cls_dict={
        "get_electron_file": (lambda self, external_files: external_files.cross_electron_trigger_sf),
        "get_electron_config": (lambda self: self.config_inst.x.cross_electron_trigger_data_effs_names),
        "weight_name": "cross_electron_trigger_data_effs",
    },
)

cross_electron_trigger_mc_effs = electron_weights.derive(
    "cross_electron_trigger_mc_effs",
    cls_dict={
        "get_electron_file": (lambda self, external_files: external_files.cross_electron_trigger_sf),
        "get_electron_config": (lambda self: self.config_inst.x.cross_electron_trigger_mc_effs_names),
        "weight_name": "cross_electron_trigger_mc_effs",
    },
)

# subclass the tau weights producer to create the tau efficiencies
cross_etau_trigger_data_effs = tau_weights.derive(
    "cross_etau_trigger_data_effs",
    cls_dict={
        "get_tau_file": (lambda self, external_files: external_files.tau_trigger_sf),
        "get_tau_config": (lambda self: self.config_inst.x.cross_etau_trigger_data_effs_names),
        "weight_name": "cross_etau_trigger_data_effs",
    },
)

cross_etau_trigger_mc_effs = tau_weights.derive(
    "cross_etau_trigger_mc_effs",
    cls_dict={
        "get_tau_file": (lambda self, external_files: external_files.tau_trigger_sf),
        "get_tau_config": (lambda self: self.config_inst.x.cross_etau_trigger_mc_effs_names),
        "weight_name": "cross_etau_trigger_mc_effs",
    },
)

# subclass the muon weights producer to create the muon efficiencies
single_muon_trigger_data_effs = muon_weights.derive(
    "single_muon_trigger_data_effs",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.muon_trigger_sf),
        "get_muon_config": (lambda self: self.config_inst.x.single_muon_trigger_data_effs_names),
        "weight_name": "single_muon_trigger_data_effs",
    },
)

single_muon_trigger_mc_effs = muon_weights.derive(
    "single_muon_trigger_mc_effs",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.muon_trigger_sf),
        "get_muon_config": (lambda self: self.config_inst.x.single_muon_trigger_mc_effs_names),
        "weight_name": "single_muon_trigger_mc_effs",
    },
)

cross_muon_trigger_data_effs = muon_weights.derive(
    "cross_muon_trigger_data_effs",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.cross_muon_trigger_sf),
        "get_muon_config": (lambda self: self.config_inst.x.cross_muon_trigger_data_effs_names),
        "weight_name": "cross_muon_trigger_data_effs",
    },
)

cross_muon_trigger_mc_effs = muon_weights.derive(
    "cross_muon_trigger_mc_effs",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.cross_muon_trigger_sf),
        "get_muon_config": (lambda self: self.config_inst.x.cross_muon_trigger_mc_effs_names),
        "weight_name": "cross_muon_trigger_mc_effs",
    },
)

# subclass the tau weights producer to create the tau efficiencies
cross_mutau_trigger_data_effs = tau_weights.derive(
    "cross_mutau_trigger_data_effs",
    cls_dict={
        "get_tau_file": (lambda self, external_files: external_files.tau_trigger_sf),
        "get_tau_config": (lambda self: self.config_inst.x.cross_mutau_trigger_data_effs_names),
        "weight_name": "cross_mutau_trigger_data_effs",
    },
)

cross_mutau_trigger_mc_effs = tau_weights.derive(
    "cross_mutau_trigger_mc_effs",
    cls_dict={
        "get_tau_file": (lambda self, external_files: external_files.tau_trigger_sf),
        "get_tau_config": (lambda self: self.config_inst.x.cross_mutau_trigger_mc_effs_names),
        "weight_name": "cross_mutau_trigger_mc_effs",
    },
)

def reshape_masked_to_oneslike_original(masked_array: ak.Array, mask: ak.Array) -> ak.Array:
    """
    Reshape a masked array to a numpy.ones_like array of the original shape.
    """
    oneslike_original = np.ones_like(mask)
    oneslike_original[mask] = masked_array
    return oneslike_original

def calculate_ditrigger_efficiency(
    single_triggered,
    cross_triggered,
    single_trigger_effs,
    cross_trigger_effs,
    cross_tau_trigger_effs_nom,
) -> ak.Array:
    """
    Calculate the combination of the single and cross trigger efficiencies.
    """

    # compute the trigger weights
    trigger_efficiency = (
        (single_trigger_effs * single_triggered) +
        (cross_tau_trigger_effs_nom * cross_trigger_effs * cross_triggered) -
        (
            single_triggered *
            cross_triggered *
            cross_tau_trigger_effs_nom *
            np.minimum(
                single_trigger_effs,
                cross_trigger_effs,
            )
        )
    )
    return trigger_efficiency


@producer(
    uses={
        "channel_id", "single_triggered", "cross_triggered",
        single_electron_trigger_data_effs, cross_electron_trigger_data_effs,
        single_electron_trigger_mc_effs, cross_electron_trigger_mc_effs,
        cross_etau_trigger_data_effs, cross_etau_trigger_mc_effs,
        single_muon_trigger_data_effs, cross_muon_trigger_data_effs,
        single_muon_trigger_mc_effs, cross_muon_trigger_mc_effs,
        cross_mutau_trigger_data_effs, cross_mutau_trigger_mc_effs,
    },
    produces={
        "etau_trigger_weight", "mutau_trigger_weight",
    } | {
        f"mutau_trigger_weight_{direction}"
        for direction in ["up", "down"]
    } | {
        f"etau_trigger_weight_{direction}"
        for direction in ["up", "down"]
    },
)
def build_trigger_weights(
    self: Producer,
    events: ak.Array,
    single_muon_triggered: ak.Array,
    cross_muon_triggered: ak.Array,
    single_electron_triggered: ak.Array,
    cross_electron_triggered: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Producer for muon trigger scale factors derived by Jona Motta. Requires external files in the
    config under ``muon_trigger_sf`` and ``cross_muon_trigger_sf``:

    .. code-block:: python

        cfg.x.external_files = DotDict.wrap({
            "muon_trigger_sf": "/afs/cern.ch/work/m/mrieger/public/mirrors/jsonpog-integration-9ea86c4c/POG/MUO/2017_UL/muon_z.json.gz",  # noqa
            "cross_muon_trigger_sf": "/afs/cern.ch/work/m/mrieger/public/mirrors/jsonpog-integration-9ea86c4c/POG/MUO/2017_UL/muon_z.json.gz",  # noqa
        })
    """

    # TODO: TOCHECK: if necessary, make it an object mask using e.g. "& events.Muon.pt > 0.0"
    etau_single_triggered = (events.channel_id == self.config_inst.channels.n.etau.id) & events.single_triggered
    mutau_single_triggered = (events.channel_id == self.config_inst.channels.n.mutau.id) & events.single_triggered
    etau_cross_triggered = (events.channel_id == self.config_inst.channels.n.etau.id) & events.cross_triggered
    mutau_cross_triggered = (events.channel_id == self.config_inst.channels.n.mutau.id) & events.cross_triggered


    # get efficiencies from the correctionlib producers
    events = self[single_muon_trigger_data_effs](events, single_muon_triggered, **kwargs)
    events = self[cross_muon_trigger_data_effs](events, cross_muon_triggered, **kwargs)
    events = self[cross_mutau_trigger_data_effs](events, cross_muon_triggered, **kwargs)
    events = self[single_electron_trigger_data_effs](events, single_electron_triggered, **kwargs)
    events = self[cross_electron_trigger_data_effs](events, cross_electron_triggered, **kwargs)
    events = self[cross_etau_trigger_data_effs](events, cross_electron_triggered, **kwargs)
    single_muon_trigger_data_efficiencies = events.single_muon_trigger_data_effs(events, single_muon_triggered, **kwargs)
    cross_muon_trigger_data_efficiencies = events.cross_muon_trigger_data_effs(events, cross_muon_triggered, **kwargs)
    cross_mutau_trigger_data_efficiencies = events.cross_mutau_trigger_data_effs(events, cross_muon_triggered, **kwargs)

    # compute the trigger weights
    muon_trigger_efficiency_data = calculate_ditrigger_efficiency(
        single_muon_triggered,
        cross_muon_triggered,
        single_muon_trigger_data_efficiencies,
        cross_muon_trigger_data_efficiencies,
        cross_mutau_trigger_data_efficiencies,
    )

    # TODO: same for MC

    # TODO: calculate SFs

    # TODO: same for electrons


    return events



@producer(
    uses={
        "channel_id", "single_triggered", "cross_triggered",
        "Electron.pt", "Electron.eta",
    },
    produces={
        "etau_trigger_weight",
    } | {
        f"etau_trigger_weight_{direction}"
        for direction in ["up", "down"]
    },
    # function to determine the correction file
    get_custom_etau_file=(lambda self, external_files: external_files.custom_etau_sf),
)
def custom_etau_trigger_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer for trigger scale factors derived by Jona Motta. Requires external files in the
    config under ``custom_etau_sf``:

    .. code-block:: python

        cfg.x.external_files = DotDict.wrap({
            "custom_etau_sf": "/afs/cern.ch/work/m/mrieger/public/mirrors/jsonpog-integration-9ea86c4c/POG/TAU/2017_UL/tau.json.gz",  # noqa
        })

    *get_custom_etau_file* can be adapted in a subclass in case it is stored differently in the external
    files. A correction set named ``"Electron-HLT-SF"`` is extracted from the etau file.

    Resources:
    https://gitlab.cern.ch/cclubbtautau/AnalysisCore/-/tree/main/data/TriggerScaleFactors/2022preEE?ref_type=heads
    """
    # get channels from the config
    ch_etau = self.config_inst.get_channel("etau")
    ch_mutau = self.config_inst.get_channel("mutau")
    ch_tautau = self.config_inst.get_channel("tautau")

    # helper to bring a flat sf array into the shape of taus, and multiply across the tau axis
    reduce_mul = lambda sf: ak.prod(layout_ak_array(sf, events.Tau.pt), axis=1, mask_identity=False)

    # the correction tool only supports flat arrays, so convert inputs to flat np view first
    pt = flat_np_view(events.Tau.pt, axis=1)
    dm = flat_np_view(events.Tau.decayMode, axis=1)

    #
    # compute nominal trigger weight
    #

    # define channel / trigger dependent masks
    channel_id = events.channel_id
    single_triggered = events.single_triggered
    dm_mask = (
        (events.Tau.decayMode == 0) |
        (events.Tau.decayMode == 1) |
        (events.Tau.decayMode == 10) |
        (events.Tau.decayMode == 11)
    )
    tautau_mask = flat_np_view(
        dm_mask & (events.Tau.pt >= 40.0) & (channel_id == ch_tautau.id),
        axis=1,
    )
    # not existing yet
    # tautauvbf_mask = flat_np_view(dm_mask & (channel_id == ch_tautau.id), axis=1)
    etau_mask = flat_np_view(
        dm_mask & (channel_id == ch_etau.id) & single_triggered & (events.Tau.pt >= 25.0),
        axis=1,
    )
    mutau_mask = flat_np_view(
        dm_mask & (channel_id == ch_mutau.id) & single_triggered & (events.Tau.pt >= 25.0),
        axis=1,
    )

    # start with flat ones
    sf_nom = np.ones_like(pt, dtype=np.float32)
    wp_config = self.config_inst.x.tau_trigger_working_points
    eval_args = lambda mask, ch, syst: (pt[mask], dm[mask], ch, wp_config.trigger_corr, "sf", syst)
    sf_nom[etau_mask] = self.trigger_corrector(*eval_args(etau_mask, "etau", "nom"))
    sf_nom[mutau_mask] = self.trigger_corrector(*eval_args(mutau_mask, "mutau", "nom"))
    sf_nom[tautau_mask] = self.trigger_corrector(*eval_args(tautau_mask, "ditau", "nom"))

    # create and store weights
    events = set_ak_column_f32(events, "tau_trigger_weight", reduce_mul(sf_nom))

    #
    # compute varied trigger weights
    #

    for direction in ["up", "down"]:
        for ch, ch_corr, mask in [
            ("etau", "etau", etau_mask),
            ("mutau", "mutau", mutau_mask),
            ("tautau", "ditau", tautau_mask),
            # ("tautauvbf", "ditauvbf", tautauvbf_mask),
        ]:
            sf_unc = sf_nom.copy()
            sf_unc[mask] = self.trigger_corrector(*eval_args(mask, ch_corr, direction))
            events = set_ak_column_f32(events, f"tau_trigger_weight_{ch}_{direction}", reduce_mul(sf_unc))

    return events


@custom_etau_trigger_weights.requires
def custom_etau_trigger_weights_requires(self: Producer, reqs: dict) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@custom_etau_trigger_weights.setup
def custom_etau_trigger_weights_setup(self: Producer, reqs: dict, inputs: dict, reader_targets: InsertableDict) -> None:
    bundle = reqs["external_files"]

    # create the trigger and id correctors
    import correctionlib
    correctionlib.highlevel.Correction.__call__ = correctionlib.highlevel.Correction.evaluate

    # load the correction set
    correction_set_etau = correctionlib.CorrectionSet.from_string(
        self.get_custom_etau_file(bundle.files).load(formatter="gzip").decode("utf-8"),
    )
    self.etau_trigger_corrector = correction_set_etau["Electron-HLT-SF"]

    # check versions
    assert self.etau_trigger_corrector.version in [0, 1]


@producer(
    uses={
        "channel_id", "single_triggered", "cross_triggered",
        "Muon.pt", "Muon.eta",
    },
    produces={
        "mutau_medium_trigger_weight",
        "mutau_tight_trigger_weight",
    } | {
        f"mutau_medium_trigger_weight_{direction}"
        for direction in ["up", "down"]
    } | {
        f"mutau_tight_trigger_weight_{direction}"
        for direction in ["up", "down"]
    },
    # function to determine the correction file
    get_custom_mutau_file=(lambda self, external_files: external_files.custom_mutau_sf),
)
def custom_mutau_trigger_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer for trigger scale factors derived by Jona Motta. Requires external files in the
    config under ``custom_mutau_sf``:

    .. code-block:: python

        cfg.x.external_files = DotDict.wrap({
            "custom_mutau_sf": "/afs/cern.ch/work/m/mrieger/public/mirrors/jsonpog-integration-9ea86c4c/POG/TAU/2017_UL/tau.json.gz",  # noqa
        })

    *get_custom_mutau_file* can be adapted in a subclass in case it is stored differently in the external
    files. Correction sets named
    ``"NUM_IsoMu20_DEN_CutBasedIdMedium_and_PFIsoMedium"`` and ``"NUM_IsoMu20_DEN_CutBasedIdTight_and_PFIsoTight"``
    are extracted from the file.

    Resources:
    https://gitlab.cern.ch/cclubbtautau/AnalysisCore/-/tree/main/data/TriggerScaleFactors/2022preEE?ref_type=heads
    """
    # get channels from the config
    ch_etau = self.config_inst.get_channel("etau")
    ch_mutau = self.config_inst.get_channel("mutau")
    ch_tautau = self.config_inst.get_channel("tautau")

    # helper to bring a flat sf array into the shape of taus, and multiply across the tau axis
    reduce_mul = lambda sf: ak.prod(layout_ak_array(sf, events.Tau.pt), axis=1, mask_identity=False)

    # the correction tool only supports flat arrays, so convert inputs to flat np view first
    pt = flat_np_view(events.Tau.pt, axis=1)
    dm = flat_np_view(events.Tau.decayMode, axis=1)

    #
    # compute nominal trigger weight
    #

    # define channel / trigger dependent masks
    channel_id = events.channel_id
    single_triggered = events.single_triggered
    dm_mask = (
        (events.Tau.decayMode == 0) |
        (events.Tau.decayMode == 1) |
        (events.Tau.decayMode == 10) |
        (events.Tau.decayMode == 11)
    )
    tautau_mask = flat_np_view(
        dm_mask & (events.Tau.pt >= 40.0) & (channel_id == ch_tautau.id),
        axis=1,
    )
    # not existing yet
    # tautauvbf_mask = flat_np_view(dm_mask & (channel_id == ch_tautau.id), axis=1)
    etau_mask = flat_np_view(
        dm_mask & (channel_id == ch_etau.id) & single_triggered & (events.Tau.pt >= 25.0),
        axis=1,
    )
    mutau_mask = flat_np_view(
        dm_mask & (channel_id == ch_mutau.id) & single_triggered & (events.Tau.pt >= 25.0),
        axis=1,
    )

    # start with flat ones
    sf_nom = np.ones_like(pt, dtype=np.float32)
    wp_config = self.config_inst.x.tau_trigger_working_points
    eval_args = lambda mask, ch, syst: (pt[mask], dm[mask], ch, wp_config.trigger_corr, "sf", syst)
    sf_nom[etau_mask] = self.trigger_corrector(*eval_args(etau_mask, "etau", "nom"))
    sf_nom[mutau_mask] = self.trigger_corrector(*eval_args(mutau_mask, "mutau", "nom"))
    sf_nom[tautau_mask] = self.trigger_corrector(*eval_args(tautau_mask, "ditau", "nom"))

    # create and store weights
    events = set_ak_column_f32(events, "tau_trigger_weight", reduce_mul(sf_nom))

    #
    # compute varied trigger weights
    #

    for direction in ["up", "down"]:
        for ch, ch_corr, mask in [
            ("etau", "etau", etau_mask),
            ("mutau", "mutau", mutau_mask),
            ("tautau", "ditau", tautau_mask),
            # ("tautauvbf", "ditauvbf", tautauvbf_mask),
        ]:
            sf_unc = sf_nom.copy()
            sf_unc[mask] = self.trigger_corrector(*eval_args(mask, ch_corr, direction))
            events = set_ak_column_f32(events, f"tau_trigger_weight_{ch}_{direction}", reduce_mul(sf_unc))

    return events


@custom_mutau_trigger_weights.requires
def custom_mutau_trigger_weights_requires(self: Producer, reqs: dict) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@custom_mutau_trigger_weights.setup
def custom_mutau_trigger_weights_setup(self: Producer, reqs: dict, inputs: dict, reader_targets: InsertableDict) -> None:
    bundle = reqs["external_files"]

    # create the trigger and id correctors
    import correctionlib
    correctionlib.highlevel.Correction.__call__ = correctionlib.highlevel.Correction.evaluate

    # load the correction set
    correction_set_mutau = correctionlib.CorrectionSet.from_string(
        self.get_custom_mutau_file(bundle.files).load(formatter="gzip").decode("utf-8"),
    )
    self.mutau_medium_trigger_corrector = correction_set_mutau["NUM_IsoMu20_DEN_CutBasedIdMedium_and_PFIsoMedium"]
    self.mutau_tight_trigger_corrector = correction_set_mutau["NUM_IsoMu20_DEN_CutBasedIdTight_and_PFIsoTight"]

    # check versions
    assert self.mutau_medium_trigger_corrector.version in [0, 1]
    assert self.mutau_tight_trigger_corrector.version in [0, 1]
