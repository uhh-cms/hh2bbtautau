import functools
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, set_ak_column
from columnflow.production.util import attach_coffea_behavior

np = maybe_import("numpy")
ak = maybe_import("awkward")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses=(
        "Electron.*", "Tau.*", "Jet.*", "HHBJet.*",
        attach_coffea_behavior,
    ),
    produces={
        "hh.*", "diTau.*", "diBJet.*",
    },
)
def hh_mass(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](
        events,
        collections={"HHBJet": {"type_name": "Jet"}},
        **kwargs,
    )

    # total number of objects per event
    n_bjets = ak.num(events.HHBJet, axis=1)
    n_taus = ak.num(events.Tau, axis=1)
    # mask to select events with exactly 2 taus
    ditau_mask = (n_taus == 2)
    diBjet_mask = (n_bjets == 2)
    dihh_mask = ditau_mask & diBjet_mask

    # four-vector sum of first two elements of each object collection (possibly fewer)
    diBJet = events.HHBJet.sum(axis=1)
    diTau = events.Tau[:, :2].sum(axis=1)
    hh = diBJet + diTau

    def save_interesting_properties(
        source: ak.Array,
        target_column: str,
        column_values: ak.Array,
        mask: ak.Array[bool],
    ):
        return set_ak_column_f32(
            source,
            target_column,
            ak.where(mask, column_values, EMPTY_FLOAT),
        )

    # write out variables to the corresponding events array, applying certain masks
    events = save_interesting_properties(events, "diBJet.mass", diBJet.mass, diBjet_mask)
    events = save_interesting_properties(events, "diBJet.eta", diBJet.eta, diBjet_mask)
    events = save_interesting_properties(events, "diBJet.pt", diBJet.pt, diBjet_mask)
    events = save_interesting_properties(events, "diTau.mass", diTau.mass, ditau_mask)
    events = save_interesting_properties(events, "diTau.eta", diTau.eta, ditau_mask)
    events = save_interesting_properties(events, "diTau.pt", diTau.pt, ditau_mask)
    events = save_interesting_properties(events, "hh.mass", hh.mass, dihh_mask)
    events = save_interesting_properties(events, "hh.eta", hh.eta, dihh_mask)
    events = save_interesting_properties(events, "hh.pt", hh.pt, dihh_mask)

    # return the events
    return events
