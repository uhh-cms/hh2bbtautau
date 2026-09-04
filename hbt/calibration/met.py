# coding: utf-8

"""
MET related calibrators.
"""

from columnflow.calibration import Calibrator, calibrator
from columnflow.production.cms.dy import recoil_corrected_met as recoil_corrected_met_prod
from columnflow.columnar_util import set_ak_column
from columnflow.util import maybe_import

ak = maybe_import("awkward")


@calibrator(
    uses={recoil_corrected_met_prod},
    produces=set(),  # set dynamically in init
)
def recoil_corrected_met(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """
    Same as :py:class:`columnflow.production.cms.dy.recoil_corrected_met`, but re-routes all produced columns to the
    original MET collection.
    """
    events = self[recoil_corrected_met_prod](events, **kwargs)

    # save uncorrected columns
    events = set_ak_column(events, f"{self.met_name}.pt_recoil_uncorrected", events[self.met_name]["pt"])
    events = set_ak_column(events, f"{self.met_name}.phi_recoil_uncorrected", events[self.met_name]["phi"])

    # overwrite with corrected columns
    events = set_ak_column(events, f"{self.met_name}.pt", events[self.corr_met_name]["pt"])
    events = set_ak_column(events, f"{self.met_name}.phi", events[self.corr_met_name]["phi"])

    # forward varied columns
    for c in self.varied_columns:
        events = set_ak_column(events, f"{self.met_name}.{c}", events[self.corr_met_name][c])

    return events


@recoil_corrected_met.init
def recoil_corrected_met_init(self: Calibrator, **kwargs) -> None:
    super(recoil_corrected_met, self).init_func(**kwargs)

    self.met_name = self.config_inst.x.met_name
    self.corr_met_name = "RecoilCorrMET"

    corr_routes = self[recoil_corrected_met_prod].produced_columns

    # add default and uncorrected columns
    self.produces.add(f"{self.met_name}.{{pt,phi}}")
    self.produces.add(f"{self.met_name}.{{pt,phi}}_recoil_uncorrected")

    # forward all up/down columns
    self.varied_columns = []
    for r in corr_routes:
        c = r[1]
        if c.endswith(("_up", "_down")):
            self.varied_columns.append(c)
            self.produces.add(f"{self.met_name}.{c}")
