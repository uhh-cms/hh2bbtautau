# coding: utf-8

"""
Special histogram producers for end-to-end DNN applications.
"""

from __future__ import annotations

import law

from columnflow.histogramming import HistProducer
from columnflow.columnar_util import full_like, EMPTY_FLOAT
from columnflow.util import maybe_import
from columnflow.types import TYPE_CHECKING, Any

from hbt.histogramming.default import default

np = maybe_import("numpy")
ak = maybe_import("awkward")
if TYPE_CHECKING:
    hist = maybe_import("hist")


# create a copy of the default hist producer and configure its fill_hist hool
e2e = default.derive("e2e")


@e2e.fill_hist
def e2e_fill_hist(self: HistProducer, h: hist.Hist, data: dict[str, Any], task: law.Task) -> None:
    # the variable field of data is supposed to be a list whose length correponds to the number of variable bins
    var_axis = h.axes[-1]
    assert var_axis.size == len(data[var_axis.name])

    # loop through bin contributions and consider them all additional weights to fill each entry
    for bin_center, bin_weights in zip(var_axis.centers, data[var_axis.name]):
        # bin weights might be empty floats in case the dnn could not evaluated for the event (e.g. wrong phase space),
        # so fill them with zero weight instead
        bin_weights = ak.where(bin_weights == EMPTY_FLOAT, 0.0, bin_weights)
        # create new data dict with the bin weight folded into the event weight and fill
        _data = {
            **data,
            var_axis.name: full_like(bin_weights, bin_center),
            "weight": data["weight"] * bin_weights,
        }
        super(e2e, self).fill_hist_func(h, _data, task=task)
