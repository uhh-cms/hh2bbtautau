# coding: utf-8

"""
Jet scale factor production.
"""

from __future__ import annotations

import functools

import law

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
