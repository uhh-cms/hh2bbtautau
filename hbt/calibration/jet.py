# coding: utf-8

"""
Custom jet energy calibration methods that disable data uncertainties (for searches).
"""

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.jets import jec, jer
from columnflow.util import maybe_import

ak = maybe_import("awkward")


# custom jec calibrator that only runs nominal correction
jec_nominal = jec.derive("jec_nominal", cls_dict={"uncertainty_sources": []})


@calibrator
def jet_energy(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """
    Common calibrator for Jet energy corrections, applying nominal JEC for data, and JEC with
    uncertainties plus JER for MC. Information about used and produced columns and dependent
    calibrators is added in a custom init function below.
    """
    if self.dataset_inst.is_mc:
        events = self[jec](events, **kwargs)
        events = self[jer](events, **kwargs)
    else:
        events = self[jec_nominal](events, **kwargs)

    return events


@jet_energy.init
def jet_energy_init(self: Calibrator) -> None:
    # add standard jec and jer for mc, and only jec nominal for dta
    if getattr(self, "dataset_inst", None):
        if self.dataset_inst.is_mc:
            self.uses |= {jec, jer}
            self.produces |= {jec, jer}
        else:
            self.uses |= {jec_nominal}
            self.produces |= {jec_nominal}
