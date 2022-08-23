# coding: utf-8

"""
Calibration methods.
"""

from columnflow.calibration import Calibrator, calibrator
from columnflow.production.seeds import deterministic_seeds
from columnflow.calibration.jets import jec, jer
from columnflow.util import maybe_import

ak = maybe_import("awkward")


# custom jec calibrator that only runs nominal correction
jec_nominal = jec.derive("jec_nominal", cls_dict={"uncertainty_sources": []})


@calibrator(
    uses={deterministic_seeds, jec_nominal, jer},
    produces={deterministic_seeds, jec_nominal, jer},
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    self[deterministic_seeds](events, **kwargs)
    self[jec_nominal](events, **kwargs)
    self[jer](events, **kwargs)

    return events
