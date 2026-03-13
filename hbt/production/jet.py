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
    named ``"jet_trigger_corrector"`` is extracted from it.

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
    lep_mask: ak.Array | None = None

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

    def __post_init__(self):
        if 0.0 < self.max_pt <= self.min_pt:
            raise ValueError(f"{self.__class__.__name__}: max_pt must be larger than min_pt")


@producer(
    uses={
        "channel_id", "VBFJet.{pt,eta}",
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
            lep_mask=None,  # mask if need to add the electron pt as variable for the correction
        )
    Resources:
    https://gitlab.cern.ch/cclubbtautau/AnalysisCore/-/blob/3e57bd1eaae7a086065c77b6c59dd6cf0600546c/data/TriggerScaleFactors/2024fullYear/VBF2tau_SF_2024.json.gz
    """
    vbf_jet_1 = events.VBFJet[jet_mask][:, 0]
    vbf_jet_2 = events.VBFJet[jet_mask][:, 1]
    # maybe better with ak.firsts(events.VBFJet.pt[jet_mask]) and ak.firsts(events.VBFJet.pt[jet_mask][1:]))
    # but should not be necessary if jet mask correctly identifies events

    variable_map = {
        "vbfjet1_pt": vbf_jet_1.pt,
        "vbfjet2_pt": vbf_jet_2.pt,
        "mjj": (vbf_jet_1 + vbf_jet_2).mass,
    }

    if self.vbfjet_config.lep_mask is not None:
        variable_map["lep_pt"] = events.Electron[self.vbfjet_config.lep_mask].pt[:, 0]

    # no efficiency needed except if triple jet trigger used, let it be decided by the config
    for syst, postfix in [
        ("nom", ""),
        ("up", "_up"),
        ("down", "_down"),
    ]:
        # get the inputs for this type of variation
        variable_map_syst = {
            **variable_map,
            "syst": syst,
        }
        inputs = [variable_map_syst[inp.name] for inp in self.vbfjet_trig_corrector.inputs]
        sf = self.vbfjet_trig_corrector(*inputs)

        # store it
        events = set_ak_column(events, f"{self.sf_name}_{postfix}", sf, value_type=np.float32)

    return events


@vbfjet_trigger_efficiencies.init
def vbfjet_trigger_efficiencies_init(self: Producer, **kwargs) -> None:
    # add the product of nominal and up/down variations to produced columns
    if self.vbfjet_config.lep_mask is not None:
        self.uses.add("Electron.pt")
    self.produces.add(f"{self.sf_name}_{{,_up,_down}}")


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

# TODO: add quadjet trigger SFs
