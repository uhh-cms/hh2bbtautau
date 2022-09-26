# coding: utf-8

"""
Calibration methods.
"""

from columnflow.calibration import Calibrator, calibrator
from columnflow.production.mc_weight import mc_weight
from columnflow.production.seeds import deterministic_seeds
from columnflow.util import maybe_import

from hbt.calibration.jet import jet_energy
from hbt.calibration.met_phi import met_phi
from hbt.calibration.tau import tec

ak = maybe_import("awkward")


@calibrator(
    uses={mc_weight, deterministic_seeds, jet_energy, met_phi, tec},
    produces={mc_weight, deterministic_seeds, jet_energy, met_phi, tec},
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    events = self[mc_weight](events, **kwargs)
    events = self[deterministic_seeds](events, **kwargs)
    events = self[jet_energy](events, **kwargs)
    events = self[met_phi](events, **kwargs)
    events = self[tec](events, **kwargs)

    return events
