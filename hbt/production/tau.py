# coding: utf-8

"""
Tau scale factor production.
"""

from __future__ import annotations

import functools

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, load_correction_set, DotDict
from columnflow.columnar_util import set_ak_column, flat_np_view, layout_ak_array
from columnflow.types import Any

from hbt.util import uppercase_wp

ak = maybe_import("awkward")
np = maybe_import("numpy")


# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={
        "channel_id",
        "Tau.{mass,pt,eta,phi,decayMode,genPartFlav}",
    },
    produces={
        "tau_weight",
    } | {
        f"tau_weight_{unc}_{{up,down}}"
        for unc in [
            "jet_stat{1,2}_dm{0,1,10,11}",
            "e_barrel", "e_endcap",
            "mu_0p0To0p4", "mu_0p4To0p8", "mu_0p8To1p2", "mu_1p2To1p7", "mu_1p7To2p3",
        ]
    },
    # only run on mc
    mc_only=True,
    # function to determine the correction file
    get_tau_file=(lambda self, external_files: external_files.tau_sf),
    # function to determine the tau tagger name
    get_tau_tagger=(lambda self: self.config_inst.x.tau_tagger),
)
def tau_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer for tau ID weights. Requires an external file in the config under ``tau_sf``:

    .. code-block:: python

        cfg.x.external_files = DotDict.wrap({
            "tau_sf": "/afs/cern.ch/work/m/mrieger/public/mirrors/jsonpog-integration-9ea86c4c/POG/TAU/2017_UL/tau.json.gz",  # noqa
        })

    *get_tau_file* can be adapted in a subclass in case it is stored differently in the external files.

    The name of the tagger should be given as an auxiliary entry in the config.

    .. code-block:: python

        cfg.x.tau_tagger = "DeepTau2018v2p5"

    *get_tau_tagger* can be adapted in a subclass in case it is stored differently in the config.

    Resources:
        - https://twiki.cern.ch/twiki/bin/view/CMS/TauIDRecommendationForRun2?rev=113
        - https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/849c6a6efef907f4033715d52290d1a661b7e8f9/POG/TAU
    """
    # get channels
    ch_etau = self.config_inst.channels.n.etau
    ch_mutau = self.config_inst.channels.n.mutau
    ch_tautau = self.config_inst.channels.n.tautau

    # get taus: one for e/mutau, two for tautau
    etau_mask = events.channel_id == ch_etau.id
    mutau_mask = events.channel_id == ch_mutau.id
    tautau_mask = events.channel_id == ch_tautau.id
    taus = ak.where(
        (etau_mask | mutau_mask),
        events.Tau[:, :1],
        ak.where(
            tautau_mask,
            events.Tau[:, :2],
            events.Tau[:, :0],
        ),
    )
    taus_flat = ak.flatten(taus, axis=1)

    # create a channel id array in the same shape of flat taus
    ch_flat = ak.where(
        etau_mask,
        [[ch_etau.id]],
        ak.where(
            mutau_mask,
            [[ch_mutau.id]],
            ak.where(
                tautau_mask,
                [2 * [ch_tautau.id]],
                [[]],
            ),
        ),
    )
    ch_flat = ak.flatten(ak.values_astype(ch_flat, np.uint8))

    # store some common values
    abseta_flat = abs(taus_flat.eta)
    dm_mask = (
        (taus_flat.decayMode == 0) |
        (taus_flat.decayMode == 1) |
        (taus_flat.decayMode == 10) |
        (taus_flat.decayMode == 11)
    )
    vs_jet_wp = uppercase_wp(self.config_inst.x.deeptau_wps.vs_jet)
    vs_e_wp = uppercase_wp(self.config_inst.x.deeptau_wps.vs_e)
    vs_mu_wp = {ch: uppercase_wp(wp) for ch, wp in self.config_inst.x.deeptau_wps.vs_mu.items()}

    # helpers to compute scale factors for various tau sources (genuine, fakes) and decay modes
    # genuine taus (separately for decay modes)
    def fill_genuine_tau(sfs_flat: np.array, syst: str, mask: np.array | ak.Array | None = None) -> None:
        genuine_mask = dm_mask & (taus_flat.genPartFlav == 5)
        if mask is not None:
            genuine_mask = genuine_mask & mask
        sfs_flat[genuine_mask] = self.id_vs_jet_corrector.evaluate(
            taus_flat.pt[genuine_mask], taus_flat.decayMode[genuine_mask], 5, vs_jet_wp, vs_e_wp, syst, "dm",
        )

    # electrons faking taus (separately for decay modes)
    def fill_e_fakes(sfs_flat: np.array, syst: str, mask: np.array | ak.Array | None = None) -> None:
        fake_mask = dm_mask & ((taus_flat.genPartFlav == 1) | (taus_flat.genPartFlav == 3))
        if mask is not None:
            fake_mask = fake_mask & mask
        sfs_flat[fake_mask] = self.id_vs_e_corrector.evaluate(
            abseta_flat[fake_mask], taus_flat.decayMode[fake_mask], taus_flat.genPartFlav[fake_mask], vs_e_wp, syst,
        )

    # muons faking taus (channel dependent)
    def fill_mu_fakes(sfs_flat: np.array, syst: str, mask: np.array | ak.Array | None = None) -> None:
        for ch in [ch_etau, ch_mutau, ch_tautau]:
            fake_mask = (ch_flat == ch.id) & ((taus_flat.genPartFlav == 2) | (taus_flat.genPartFlav == 4))
            if mask is not None:
                fake_mask = fake_mask & mask
            sfs_flat[fake_mask] = self.id_vs_mu_corrector.evaluate(
                abseta_flat[fake_mask], taus_flat.genPartFlav[fake_mask], vs_mu_wp[ch.name], syst,
            )

    # helper to reshape sfs_flat to the shape of taus, multiply across tau axis and store the results
    def add_weight(events: ak.Array, weight_name: str, sfs_flat: np.array) -> ak.Array:
        sfs = layout_ak_array(sfs_flat, taus.pt)
        events = set_ak_column_f32(events, weight_name, ak.prod(sfs, axis=1, mask_identity=False))
        return events

    # prepare per-tau scale factors, starting with ones, then fill values for specific tau sources
    sfs_flat = np.ones(len(taus_flat), dtype=np.float32)
    fill_genuine_tau(sfs_flat, "nom")
    fill_e_fakes(sfs_flat, "nom")
    fill_mu_fakes(sfs_flat, "nom")
    events = add_weight(events, "tau_weight", sfs_flat)

    # variations
    for direction in ["up", "down"]:
        # genuine taus
        for dm in [0, 1, 10, 11]:
            for i in range(2):
                _sfs_flat = sfs_flat.copy()
                fill_genuine_tau(_sfs_flat, f"stat{i + 1}_dm{dm}_{direction}", mask=(taus_flat.decayMode == dm))
                events = add_weight(events, f"tau_weight_jet_stat{i + 1}_dm{dm}_{direction}", _sfs_flat)

        # electron fakes
        for region_name, region_mask in [
            ("barrel", (abseta_flat < 1.5)),
            ("endcap", (abseta_flat >= 1.5)),
        ]:
            _sfs_flat = sfs_flat.copy()
            fill_e_fakes(_sfs_flat, direction, mask=region_mask)
            events = add_weight(events, f"tau_weight_e_{region_name}_{direction}", _sfs_flat)

        # muon fakes
        for region_name, region_mask in [
            ("0p0To0p4", (abseta_flat < 0.4)),
            ("0p4To0p8", ((abseta_flat >= 0.4) & (abseta_flat < 0.8))),
            ("0p8To1p2", ((abseta_flat >= 0.8) & (abseta_flat < 1.2))),
            ("1p2To1p7", ((abseta_flat >= 1.2) & (abseta_flat < 1.7))),
            ("1p7To2p3", (abseta_flat >= 1.7)),
        ]:
            _sfs_flat = sfs_flat.copy()
            fill_mu_fakes(_sfs_flat, direction, mask=region_mask)
            events = add_weight(events, f"tau_weight_mu_{region_name}_{direction}", _sfs_flat)

    return events


@tau_weights.requires
def tau_weights_requires(self: Producer, task: law.Task, reqs: dict, **kwargs) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(task)


@tau_weights.setup
def tau_weights_setup(
    self: Producer,
    task: law.Task,
    reqs: dict[str, DotDict[str, Any]],
    **kwargs,
) -> None:
    # create the trigger and id correctors
    tau_file = self.get_tau_file(reqs["external_files"].files)
    correction_set = load_correction_set(tau_file)
    tagger_name = self.get_tau_tagger()
    self.id_vs_jet_corrector = correction_set[f"{tagger_name}VSjet"]
    self.id_vs_e_corrector = correction_set[f"{tagger_name}VSe"]
    self.id_vs_mu_corrector = correction_set[f"{tagger_name}VSmu"]

    # check versions
    assert self.id_vs_jet_corrector.version in {1, 2, 3}
    assert self.id_vs_e_corrector.version in {1}
    assert self.id_vs_mu_corrector.version in {1}


@producer(
    uses={
        "channel_id", "single_triggered", "cross_triggered", "matched_trigger_ids",
        "Tau.{pt,decayMode}",
    },
    produces={
        "tau_trigger_eff_{data,mc}_{etau,mutau,tautau,tautaujet}",
        "tau_trigger_eff_{data,mc}_{etau,mutau,tautau,tautaujet}_dm{0,1,10,11}_{up,down}",
    },
    # only run on mc
    mc_only=True,
    # function to determine the correction file
    get_tau_file=(lambda self, external_files: (external_files.tau_trigger_sf if self.config_inst.campaign.x.year == 2024 else external_files.tau_sf)), # noqa E501
    get_tau_corrector=(lambda self: self.config_inst.x.tau_trigger_corrector),
)
def tau_trigger_efficiencies(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer for trigger scale factors derived by the TAU POG at object level. Requires an external file in the
    config under ``tau_sf``:

    .. code-block:: python

        cfg.x.external_files = DotDict.wrap({
            "tau_sf": "/afs/cern.ch/work/m/mrieger/public/mirrors/jsonpog-integration-9ea86c4c/POG/TAU/2017_UL/tau.json.gz",  # noqa
        })

    *get_tau_file* can be adapted in a subclass in case it is stored differently in the external
    files. A correction set named ``"tau_trigger"`` is extracted from it.

    Resources:
    https://twiki.cern.ch/twiki/bin/view/CMS/TauIDRecommendationForRun2?rev=113
    https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/849c6a6efef907f4033715d52290d1a661b7e8f9/POG/TAU
    """
    # get channels from the config
    ch_etau = self.config_inst.channels.n.etau
    ch_mutau = self.config_inst.channels.n.mutau
    ch_tautau = self.config_inst.channels.n.tautau

    # find out which triggers are passed
    cross_lt_trigger_passed = ak.zeros_like(events.channel_id, dtype=bool)
    tautau_trigger_passed = ak.zeros_like(events.channel_id, dtype=bool)
    tautaujet_trigger_passed = ak.zeros_like(events.channel_id, dtype=bool)
    tautauvbf_trigger_passed = ak.zeros_like(events.channel_id, dtype=bool)
    for trigger in self.config_inst.x.triggers:
        if trigger.has_tag("cross_e_tau") or trigger.has_tag("cross_mu_tau"):
            cross_lt_trigger_passed = (
                cross_lt_trigger_passed |
                np.any(events.matched_trigger_ids == trigger.id, axis=-1)
            )
        if trigger.has_tag("cross_tau_tau"):
            tautau_trigger_passed = (
                tautau_trigger_passed |
                np.any(events.matched_trigger_ids == trigger.id, axis=-1)
            )
        if trigger.has_tag("cross_tau_tau_jet"):
            tautaujet_trigger_passed = (
                tautaujet_trigger_passed |
                np.any(events.matched_trigger_ids == trigger.id, axis=-1)
            )
        if trigger.has_tag("cross_tau_tau_vbf"):
            tautauvbf_trigger_passed = (
                tautauvbf_trigger_passed |
                np.any(events.matched_trigger_ids == trigger.id, axis=-1)
            )

    # the correction tool only supports flat arrays, so convert inputs to flat np view first
    pt = flat_np_view(events.Tau.pt, axis=1)
    dm = flat_np_view(events.Tau.decayMode, axis=1)

    #
    # compute nominal trigger weight
    #

    # define channel / trigger dependent masks
    channel_id = events.channel_id

    default_tautau_mask = (
        (channel_id == ch_tautau.id) &
        ((ak.local_index(events.Tau) == 0) | (ak.local_index(events.Tau) == 1))
    )

    # TODO: add additional phase space requirements for tautauvbf
    tautau_mask = default_tautau_mask & tautau_trigger_passed
    flat_tautau_mask = flat_np_view(tautau_mask, axis=1)
    tautaujet_mask = default_tautau_mask & tautaujet_trigger_passed
    flat_tautaujet_mask = flat_np_view(tautaujet_mask, axis=1)

    # not existing yet
    # tautauvbf_mask = flat_np_view(default_tautau_mask & tautauvbf_trigger_passed, axis=1)
    etau_mask = (channel_id == ch_etau.id) & cross_lt_trigger_passed & (ak.local_index(events.Tau) == 0)
    flat_etau_mask = flat_np_view(etau_mask, axis=1)

    mutau_mask = (channel_id == ch_mutau.id) & cross_lt_trigger_passed & (ak.local_index(events.Tau) == 0)
    flat_mutau_mask = flat_np_view(mutau_mask, axis=1)

    # start with flat ones
    for kind in ["data", "mc"]:
        wp_config = self.config_inst.x.tau_trigger_working_points
        eval_args = lambda mask, ch, syst: (pt[mask], dm[mask], ch, wp_config.trigger_corr, f"eff_{kind}", syst)
        for corr_channel in ["etau", "mutau", "tautau", "tautaujet"]:  # TODO: add tautauvbf
            if corr_channel == "etau":
                mask = flat_etau_mask
                corr_channel_arg = corr_channel
            elif corr_channel == "mutau":
                mask = flat_mutau_mask
                corr_channel_arg = corr_channel
            elif corr_channel == "tautau":
                mask = flat_tautau_mask
                corr_channel_arg = "ditau"
            elif corr_channel == "tautaujet":
                mask = flat_tautaujet_mask
                corr_channel_arg = "ditaujet"
            else:
                raise ValueError(f"Unknown channel {corr_channel}")
            sf_nom = np.ones_like(pt, dtype=np.float32)
            sf_nom[mask] = self.tau_trig_corrector.evaluate(*eval_args(mask, corr_channel_arg, "nom"))
            # create and store weights
            events = set_ak_column_f32(
                events,
                f"tau_trigger_eff_{kind}_{corr_channel}",
                layout_ak_array(sf_nom, events.Tau.pt),
            )

        #
        # compute varied trigger weights
        #

        for ch, ch_corr, mask in [
            ("etau", "etau", etau_mask),
            ("mutau", "mutau", mutau_mask),
            ("tautau", "ditau", tautau_mask),
            ("tautaujet", "ditaujet", tautaujet_mask),
            # ("tautauvbf", "ditauvbf", tautauvbf_mask),
        ]:
            for decay_mode in [0, 1, 10, 11]:
                decay_mode_mask = mask & (events.Tau.decayMode == decay_mode)
                flat_decay_mode_mask = flat_np_view(decay_mode_mask, axis=1)
                for direction in ["up", "down"]:
                    # only possible with object-level information
                    sf_unc_flat = flat_np_view(events[f"tau_trigger_eff_{kind}_{ch}"], copy=True)
                    sf_unc_flat[flat_decay_mode_mask] = self.tau_trig_corrector.evaluate(
                        *eval_args(flat_decay_mode_mask, ch_corr, direction),
                    )
                    events = set_ak_column_f32(
                        events,
                        f"tau_trigger_eff_{kind}_{ch}_dm{decay_mode}_{direction}",
                        layout_ak_array(sf_unc_flat, events[f"tau_trigger_eff_{kind}_{ch}"]),
                    )

    return events


@tau_trigger_efficiencies.requires
def tau_trigger_efficiencies_requires(self: Producer, task: law.Task, reqs: dict, **kwargs) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(task)


@tau_trigger_efficiencies.setup
def tau_trigger_efficiencies_setup(
    self: Producer,
    task: law.Task,
    reqs: dict[str, DotDict[str, Any]],
    **kwargs,
) -> None:
    # create the trigger and id correctors
    tau_file = self.get_tau_file(reqs["external_files"].files)
    corrector_name = self.get_tau_corrector()
    self.tau_trig_corrector = load_correction_set(tau_file)[corrector_name]

    # check versions
    assert self.tau_trig_corrector.version in [0, 1]
