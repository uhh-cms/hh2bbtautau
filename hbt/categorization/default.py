from columnflow.categorization import Categorizer, categorizer
from columnflow.util import maybe_import

ak = maybe_import("awkward")


@categorizer(
    uses={"event"},
    exposed=True,
)
def cat_incl(self: Categorizer, events: ak.Array, **kwargs) -> ak.Array:
    return events, ak.ones_like(events.event)


@categorizer(
    uses={"Jet"},
    exposed=True,
)
def cat_2j(self: Categorizer, events: ak.Array, **kwargs) -> ak.Array:
    return events, ak.num(events.Jet, axis=-1) == 2
