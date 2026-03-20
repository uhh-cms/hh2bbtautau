# coding: utf-8

"""
Jet scale factor production.
"""

from __future__ import annotations

import functools

import law

import dataclasses

from columnflow.production import Producer, producer
from columnflow.columnar_util import set_ak_column
from columnflow.util import maybe_import, load_correction_set

ak = maybe_import("awkward")
np = maybe_import("numpy")


# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={
        "channel_id", "HHBJet.{pt,eta}",
    },
    # only run on mc
    mc_only=True,
    # function to determine the correction file
    get_jet_file=(lambda self, external_files: external_files.trigger_sf.ditau_jet),
    get_jet_corrector=(lambda self: self.config_inst.x.jet_trigger_corrector),
    efficiency_name="jet_trigger_eff",
)
def jet_trigger_efficiencies(
    self: Producer,
    events: ak.Array,
    jet_mask: ak.Array | type(Ellipsis) = Ellipsis,
    **kwargs,
) -> ak.Array:
    """
    Producer for jet trigger efficiencies derived by the CCLUB group at object level. Requires an external file in the
    config under ``trigger_sf.ditau_jet``.

    *get_jet_file* can be adapted in a subclass in case it is stored differently in the external files. A correction set
    named after the ``jet_trigger_corrector`` entry in the config is extracted from it.

    Resources:
    https://gitlab.cern.ch/cclubbtautau/AnalysisCore/-/blob/59ae66c4a39d3e54afad5733895c33b1fb511c47/data/TriggerScaleFactors/2023postBPix/ditaujet_jetleg_SFs_postBPix.json
    """
    variable_map = {
        "pt": events.HHBJet.pt[jet_mask],
        "abseta": abs(events.HHBJet.eta[jet_mask]),
    }

    # loop over efficiency type
    for kind in ["data", "mc"]:
        # loop over systematics
        for syst, postfix in [
            ("nom", ""),
            ("up", "_up"),
            ("down", "_down"),
        ]:
            # get the inputs for this type of variation
            variable_map_syst = {
                **variable_map,
                "syst": syst,
                "corrtype": kind,
                "syst_var": "leading_jet_pt_nom",
                # TODO: check if other variations needed, e.g. "leading_jet_pt_nomJer_Total_up"
            }
            inputs = [variable_map_syst[inp.name] for inp in self.jet_trig_corrector.inputs]
            sf = self.jet_trig_corrector(*inputs)

            # store it
            events = set_ak_column(events, f"{self.efficiency_name}_{kind}{postfix}", sf, value_type=np.float32)

    return events


@jet_trigger_efficiencies.init
def jet_trigger_efficiencies_init(self: Producer, **kwargs) -> None:
    # add the product of nominal and up/down variations to produced columns
    self.produces.add(f"{self.efficiency_name}_{{data,mc}}{{,_up,_down}}")


@jet_trigger_efficiencies.requires
def jet_trigger_efficiencies_requires(self: Producer, task: law.Task, reqs: dict) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(task)


@jet_trigger_efficiencies.setup
def jet_trigger_efficiencies_setup(
    self: Producer,
    task: law.Task,
    reqs: dict,
    inputs: dict,
    reader_targets: law.util.InsertableDict,
) -> None:
    bundle = reqs["external_files"]

    # create the trigger and id correctors
    correction_set = load_correction_set(self.get_jet_file(bundle.files))
    self.jet_trig_corrector = correction_set[self.get_jet_corrector()]

    # check versions
    assert self.jet_trig_corrector.version in {0, 1, 2}


@producer(
    jet_name="Jet",
    exposed=False,
)
def jet_multiplicity(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    return ak.num(events[self.jet_name], axis=-1)


@jet_multiplicity.init
def jet_multiplicity_init(self: Producer) -> None:
    self.uses.add(f"{self.jet_name}.pt")


@producer(
    jet_name="Jet",
    exposed=False,
)
def bjet_multiplicity(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    return ak.sum(events[self.jet_name][self.btag_column] > self.btag_wp, axis=-1)


@bjet_multiplicity.init
def bjet_multiplicity_init(self: Producer) -> None:
    self.btag_column = self.config_inst.x.btag_default.jet_column
    self.btag_wp = self.config_inst.x.btag_default.wp

    self.uses.add(f"{self.jet_name}.{self.btag_column}")


@dataclasses.dataclass
class VBFjetSFConfig:
    correction: str
    corr_type: str = ""
    lep_used: bool = False  # for now only vbf e concerned, additional entries needed if others are too

    @classmethod
    def new(cls, obj: VBFjetSFConfig | tuple[str, str]) -> VBFjetSFConfig:
        # purely for backwards compatibility with the old tuple format
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, str):
            return cls(obj)
        if isinstance(obj, (list, tuple)):
            return cls(*obj)
        if isinstance(obj, dict):
            return cls(**obj)
        raise ValueError(f"cannot convert {obj} to VBFjetSFConfig")


@producer(
    uses={
        "channel_id", "VBFJet.{pt,eta,phi,mass}",
    },
    # only run on mc
    mc_only=True,
    # function to determine the correction file
    get_vbfjet_file=(lambda self, external_files: external_files.trigger_sf.vbf_ditau),
    get_vbfjet_config=(lambda self: self.config_inst.x.vbfjet_ditau_trigger_config),
    sf_name="vbfjet_trigger_sf",
)
def vbfjet_trigger_efficiencies(
    self: Producer,
    events: ak.Array,
    jet_mask: ak.Array | type(Ellipsis) = Ellipsis,
    **kwargs,
) -> ak.Array:
    """
    Producer for jet trigger efficiencies derived by the CCLUB group at object level. Requires an external file in the
    config under ``trigger_sf.vbf_ditau``.

    *get_jet_file* can be adapted in a subclass in case it is stored differently in the external files.

    The name of the correction set and the type of efficiency/sf string for the weight evaluation should
    be given as an auxiliary entry in the config:

    .. code-block:: python

        cfg.x.vbfjet_trigger_config = VBFjetSFConfig(
            correction="VBFtrigSF",
            corr_type="sf",  # or "eff_mc", "eff_data" if only efficiencies should be applied
            lep_used=False,  # set to True if need to add the electron pt as variable for the correction
        )
    Resources:
    https://gitlab.cern.ch/cclubbtautau/AnalysisCore/-/blob/3e57bd1eaae7a086065c77b6c59dd6cf0600546c/data/TriggerScaleFactors/2024fullYear/VBF2tau_SF_2024.json.gz
    """
    if jet_mask is Ellipsis:
        jet_mask = np.ones_like(events.VBFJet.pt, dtype=bool)

    # pt sorting
    vbfjet_pt_sorting = ak.argsort(events.VBFJet.pt[jet_mask], axis=-1, ascending=False)
    vbf_jet_1 = ak.firsts(events.VBFJet[vbfjet_pt_sorting][jet_mask[vbfjet_pt_sorting]][:, :1], axis=1)
    vbf_jet_2 = ak.firsts(events.VBFJet[vbfjet_pt_sorting][jet_mask[vbfjet_pt_sorting]][:, 1:2], axis=1)

    variable_map = {
        "vbfjet1_pt": vbf_jet_1.pt,
        "vbfjet2_pt": vbf_jet_2.pt,
        "mjj": (vbf_jet_1 + vbf_jet_2).mass,
    }

    # check that the vbfjet1 and vbfjet2 are always there for the same events
    if ak.any(ak.is_none(vbf_jet_2.pt[~ak.is_none(vbf_jet_1.pt)])) or ak.any(ak.is_none(vbf_jet_1.pt[~ak.is_none(vbf_jet_2.pt)])):  # noqa: E501
        raise ValueError("vbfjet2 is None while vbfjet1 is not, check jet mask and sorting")

    if self.vbfjet_config.lep_used:
        etau_channel_id = self.config_inst.channels.n.etau.id
        cross_vbf_e_trigger_passed = ak.zeros_like(events.channel_id, dtype=bool)

        # verify triggered by correct trigger
        for trigger in self.config_inst.x.triggers:
            if trigger.has_tag("cross_e_vbf"):
                cross_vbf_e_trigger_passed = cross_vbf_e_trigger_passed | np.any(events.matched_trigger_ids == trigger.id, axis=-1)  # noqa

        mask = (
            (events.channel_id == etau_channel_id) &
            cross_vbf_e_trigger_passed &
            (events.Electron.pt > 0)
        )
        variable_map["lep_pt"] = ak.firsts(events.Electron[mask][:, :1].pt, axis=1)

        # check that the electron is always there for the same events as the vbf jets
        if ak.any(ak.is_none(variable_map["lep_pt"][~ak.is_none(vbf_jet_1.pt)])) or ak.any(ak.is_none(vbf_jet_1.pt[~ak.is_none(variable_map["lep_pt"])])):  # noqa: E501
            raise ValueError("electron is None while vbfjet1 is not, check trigger matching and sorting")

    # no efficiency needed except if triple jet trigger used, let it be decided by the config
    for syst, postfix in [
        ("nom", ""),
        ("up", "_up"),
        ("down", "_down"),
    ]:
        # get the inputs for this type of variation
        variable_map_syst = {
            **variable_map,
            "corr_type": self.vbfjet_config.corr_type,
            "syst": syst,
        }
        inputs = [variable_map_syst[inp.name] for inp in self.vbfjet_trig_corrector.inputs]
        sf = self.vbfjet_trig_corrector(*inputs)

        # check whether any selected event gets None
        event_mask = ak.fill_none(ak.sum(jet_mask, axis=-1) == 2, False)
        if ak.any(ak.is_none(sf[event_mask])):
            raise ValueError("None value in vbfjet trigger sf, check inputs and correction file")

        # maybe TODO: remove Nones? should never be used anyway since they are for events not passing the triggers
        # sf = ak.fill_none(sf, 1.0)

        # inflate uncertainty bei 15% to account for JES
        if syst == "up":
            sf = (
                events[f"{self.sf_name}"] +
                np.sqrt(
                    np.power(sf - events[f"{self.sf_name}"], 2) +
                    np.power(events[f"{self.sf_name}"] * 0.15, 2),
                )
            )
        if syst == "down":
            sf = (
                events[f"{self.sf_name}"] -
                np.sqrt(
                    np.power(sf - events[f"{self.sf_name}"], 2) +
                    np.power(events[f"{self.sf_name}"] * 0.15, 2),
                )
            )

        # store it
        events = set_ak_column(events, f"{self.sf_name}{postfix}", sf, value_type=np.float32)

    return events


@vbfjet_trigger_efficiencies.init
def vbfjet_trigger_efficiencies_init(self: Producer, **kwargs) -> None:
    # add the product of nominal and up/down variations to produced columns
    if self.get_vbfjet_config().lep_used:
        self.uses.add("Electron.pt")
        self.uses.add("matched_trigger_ids")
    self.produces.add(f"{self.sf_name}{{,_up,_down}}")


@vbfjet_trigger_efficiencies.requires
def vbfjet_trigger_efficiencies_requires(self: Producer, task: law.Task, reqs: dict) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(task)


@vbfjet_trigger_efficiencies.setup
def vbfjet_trigger_efficiencies_setup(
    self: Producer,
    task: law.Task,
    reqs: dict,
    inputs: dict,
    reader_targets: law.util.InsertableDict,
) -> None:
    bundle = reqs["external_files"]

    # create the trigger and id correctors
    correction_set = load_correction_set(self.get_vbfjet_file(bundle.files))

    self.vbfjet_config = self.get_vbfjet_config()
    self.vbfjet_trig_corrector = correction_set[self.vbfjet_config.correction]

    # check versions
    assert self.vbfjet_trig_corrector.version in {0, 1}


@producer(
    uses={
        "channel_id", "HHBJet.{pt,eta}",
    },
    # only run on mc
    mc_only=True,
    # function to determine the correction file
    get_quadjet_jet_file=(lambda self, external_files: external_files.trigger_sf.quadjet_jet),
    get_quadjet_jet_corrector=(lambda self: self.config_inst.x.jet_quadjet_trigger_corrector),
    sf_name="quadjet_jet_trigger_sf",
)
def quadjet_jet_trigger_sf(
    self: Producer,
    events: ak.Array,
    jet_mask: ak.Array | type(Ellipsis) = Ellipsis,
    **kwargs,
) -> ak.Array:
    """
    Producer for quadjet jet trigger efficiencies derived by the CCLUB group at object level.
    Requires an external file in the config under ``trigger_sf.quadjet_jet``.

    *get_quadjet_jet_file* can be adapted in a subclass in case it is stored differently in the external files.
    A correction set named after the ``jet_quadjet_trigger_corrector`` entry in the config is extracted from it.

    Resources:
    https://gitlab.cern.ch/cclubbtautau/AnalysisCore/-/blob/3e57bd1eaae7a086065c77b6c59dd6cf0600546c/data/TriggerScaleFactors/2024fullYear/ParkingHH_PNet1BTag0p20_BTag.json.gz
    """
    if jet_mask is Ellipsis:
        jet_mask = np.ones_like(events.HHBJet.pt, dtype=bool)
    jet_pt_sorting = ak.argsort(events.HHBJet.pt[jet_mask], axis=-1, ascending=False)
    hhbjet_1 = ak.firsts(events.HHBJet[jet_pt_sorting][jet_mask[jet_pt_sorting]][:, :1], axis=1)

    variable_map = {
        "bjet1_btag": hhbjet_1[f"{self.config_inst.x.btag_default.jet_column}"],
    }

    for syst, postfix in [
        ("nom", ""),
        ("up", "_up"),
        ("down", "_down"),
    ]:
        # get the inputs for this type of variation
        variable_map_syst = {
            **variable_map,
            "corr_type": "sf",
            "syst": syst,
        }
        inputs = [variable_map_syst[inp.name] for inp in self.quadjet_trig_corrector.inputs]
        sf = self.quadjet_trig_corrector(*inputs)

        # store it
        events = set_ak_column(events, f"{self.sf_name}{postfix}", sf, value_type=np.float32)

    return events


@quadjet_jet_trigger_sf.init
def quadjet_jet_trigger_sf_init(self: Producer, **kwargs) -> None:
    # add the product of nominal and up/down variations to produced columns
    self.uses.add(f"HHBJet.{self.config_inst.x.btag_default.jet_column}")
    self.produces.add(f"{self.sf_name}{{,_up,_down}}")


@quadjet_jet_trigger_sf.requires
def quadjet_jet_trigger_sf_requires(self: Producer, task: law.Task, reqs: dict) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(task)


@quadjet_jet_trigger_sf.setup
def quadjet_jet_trigger_sf_setup(
    self: Producer,
    task: law.Task,
    reqs: dict,
    inputs: dict,
    reader_targets: law.util.InsertableDict,
) -> None:
    bundle = reqs["external_files"]

    # create the trigger and id correctors
    correction_set = load_correction_set(self.get_quadjet_jet_file(bundle.files))
    self.quadjet_trig_corrector = correction_set[self.get_quadjet_jet_corrector()]

    # check versions
    assert self.quadjet_trig_corrector.version in {0, 1, 2}
