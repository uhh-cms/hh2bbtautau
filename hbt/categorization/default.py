<<<<<<< HEAD
# coding: utf-8

"""
Exemplary selection methods.
"""

from columnflow.categorization import Categorizer, categorizer
from columnflow.util import maybe_import


ak = maybe_import("awkward")


#
# categorizer functions used by categories definitions
#

@categorizer(uses={"event"})
def cat_incl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # fully inclusive selection
    return events, ak.ones_like(events.event) == 1


@categorizer(uses={"Jet.pt"})
def cat_2j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # two or more jets
    return events, ak.num(events.Jet.pt, axis=1) >= 2
=======
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
>>>>>>> abe5c79788e297b200656acf57c0d2f7d75e491f
