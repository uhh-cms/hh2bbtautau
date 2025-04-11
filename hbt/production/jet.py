# coding: utf-8

"""
Jet scale factor production.
"""

from __future__ import annotations

import functools

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, load_correction_set
from columnflow.columnar_util import set_ak_column, flat_np_view, layout_ak_array


ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={
        "channel_id", "HHBJet.{pt,eta}",
    },
    produces={
        "ditaujet_trigger_jet_weight_eff_{data,mc}{,_up,_down}",
    },
    # only run on mc
    mc_only=True,
    # function to determine the correction file
    get_jet_file=(lambda self, external_files: external_files.jet_trigger_sf),
    get_jet_corrector=(lambda self: self.config_inst.x.jet_trigger_corrector),
    weight_name="ditaujet_trigger_jet_weight",
)
def jet_trigger_weights(
    self: Producer,
    events: ak.Array,
    jet_mask: ak.Array | type(Ellipsis) = Ellipsis,
    **kwargs,
) -> ak.Array:
    """
    Producer for trigger scale factors derived by the CCLUB group. Requires an external file in the
    config under ``jet_trigger_sf``:

    .. code-block:: python

        cfg.x.external_files = DotDict.wrap({
            "jet_trigger_sf": (f"{trigger_json_mirror}/{cclub_eras}/ditaujet_jetleg_SFs_{jet_tag}.json", "v1"),  # noqa
        })

    *get_jet_file* can be adapted in a subclass in case it is stored differently in the external
    files. A correction set named ``"jet_trigger_corrector"`` is extracted from it.

    Resources:
    https://gitlab.cern.ch/cclubbtautau/AnalysisCore/-/blob/59ae66c4a39d3e54afad5733895c33b1fb511c47/data/TriggerScaleFactors/2023postBPix/ditaujet_jetleg_SFs_postBPix.json
    """

    # flat absolute eta and pt views
    abs_eta = flat_np_view(abs(events.HHBJet.eta[jet_mask]), axis=1)
    pt = flat_np_view(events.HHBJet.pt[jet_mask], axis=1)

    variable_map = {
        "pt": pt,
        "abseta": abs_eta,
    }

    # loop over efficiency type
    for corrtype_name, corrtype in [("data", "eff_data"), ("mc", "eff_mc")]:
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
                "data_or_mc": corrtype_name,
            }
            inputs = [variable_map_syst[inp.name] for inp in self.jet_trig_corrector.inputs]
            sf_flat = self.jet_trig_corrector(*inputs)

            # add the correct layout to it
            sf = layout_ak_array(sf_flat, events.HHBJet.pt[jet_mask])

            # create the product over all muons in one event
            weight = ak.prod(sf, axis=1, mask_identity=False)

            # store it
            events = set_ak_column(events, f"{self.weight_name}_{corrtype}{postfix}", weight, value_type=np.float32)

    return events


@jet_trigger_weights.requires
def jet_trigger_weights_requires(self: Producer, task: law.Task, reqs: dict) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(task)


@jet_trigger_weights.setup
def jet_trigger_weights_setup(
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
    assert self.jet_trig_corrector.version in [0, 1]
