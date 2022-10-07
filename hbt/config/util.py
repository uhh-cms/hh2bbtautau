# coding: utf-8

"""
Config-related object definitions and utils.
"""

from __future__ import annotations

from order import UniqueObject, TagMixin, DataSourceMixin
from order.util import typed


class TriggerLeg(object):
    """
    Container class storing information about trigger legs:

        - *pdg_id*: The id of the object that should have caused the trigger leg to fire.
        - *min_pt*: The minimum transverse momentum in GeV of the triggered object.
        - *trigger_bits*: Integer bit mask or masks describing whether the last filter of a trigger fired.
          See https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py.
          Per mask, any of the bits should match (*OR*). When multiple masks are configured, each of
          them should match (*AND*).

    For accepted types and conversions, see the *typed* setters implemented in this class.
    """

    def __init__(self, pdg_id=None, min_pt=None, trigger_bits=None):
        super().__init__()

        # instance members
        self._pdg_id = None
        self._min_pt = None
        self._trigger_bits = None

        # set initial values
        self.pdg_id = pdg_id
        self.min_pt = min_pt
        self.trigger_bits = trigger_bits

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} "
            f"'pdg_id={self.pdg_id}, min_pt={self.min_pt}, trigger_bits={self.trigger_bits}' "
            f"at {hex(id(self))}>"
        )

    @typed
    def pdg_id(self, pdg_id: int | None) -> int | None:
        if pdg_id is None:
            return None

        if not isinstance(pdg_id, int):
            raise TypeError(f"invalid pdg_id: {pdg_id}")

        return pdg_id

    @typed
    def min_pt(self, min_pt: int | float | None) -> float | None:
        if min_pt is None:
            return None

        if isinstance(min_pt, int):
            min_pt = float(min_pt)
        if not isinstance(min_pt, float):
            raise TypeError(f"invalid min_pt: {min_pt}")

        return min_pt

    @typed
    def trigger_bits(
        self,
        trigger_bits: int | tuple[int] | list[int] | None,
    ) -> list[int] | None:
        if trigger_bits is None:
            return None

        # cast to list
        if isinstance(trigger_bits, tuple):
            trigger_bits = list(trigger_bits)
        elif not isinstance(trigger_bits, list):
            trigger_bits = [trigger_bits]

        # check bit types
        for bit in trigger_bits:
            if not isinstance(bit, int):
                raise TypeError(f"invalid trigger bit: {bit}")

        return trigger_bits


class Trigger(UniqueObject, TagMixin, DataSourceMixin):
    """
    Container class storing information about triggers:

        - *name*: The path name of a trigger that should have fired.
        - *id*: A unique id of the trigger.
        - *run_range*: An inclusive range describing the runs where the trigger is to be applied
          (usually only defined by data).
        - *legs*: A list of :py:class:`TriggerLeg` objects contraining additional information and
          constraints of particular trigger legs.

    For accepted types and conversions, see the *typed* setters implemented in this class.

    In addition, two base classes from *order* provide additional functionality via mixins:

        - *tags*: Trigger objects can be assigned *tags* that can be checked later on, e.g. to
          describe the type of the trigger ("single_mu", "cross", ...).
        - *is_data*: A flag denoting whether the trigger is only meant to be applied on data
          (*True*), mc (*False*) or on both (*None*).
    """

    allow_undefined_data_source = True

    def __init__(self, name, id, run_range=None, legs=None, tags=None, is_data=None):
        UniqueObject.__init__(self, name, id)
        TagMixin.__init__(self, tags=tags)
        DataSourceMixin.__init__(self, is_data=is_data)

        # instance members
        self._run_range = None
        self._leg = None

        # set initial values
        self.run_range = run_range
        self.legs = legs

    def __repr__(self):
        data_source = "" if self.data_source is None else f", {self.data_source}-only"
        return (
            f"<{self.__class__.__name__} 'name={self.name}, nlegs={self.n_legs}{data_source}' "
            f"at {hex(id(self))}>"
        )

    @typed
    def name(self, name: str) -> str:
        if not isinstance(name, str):
            raise TypeError(f"invalid name: {name}")
        if not name.startswith("HLT_"):
            raise ValueError(f"invalid name: {name}")

        return name

    @typed
    def run_range(
        self,
        run_range: tuple[int] | list[int] | None,
    ) -> tuple[int] | None:
        if run_range is None:
            return None

        # cast list to tuple
        if isinstance(run_range, list):
            run_range = tuple(run_range)

        # run_range must be a tuple with to integers
        if not isinstance(run_range, tuple):
            raise TypeError(f"invalid run_range: {run_range}")
        if len(run_range) != 2:
            raise ValueError(f"invalid run_range length: {run_range}")
        if not isinstance(run_range[0], int):
            raise ValueError(f"invalid run_range start: {run_range[0]}")
        if not isinstance(run_range[1], int):
            raise ValueError(f"invalid run_range end: {run_range[1]}")

        return run_range

    @typed
    def legs(
        self,
        legs: (
            dict |
            tuple[dict] |
            list[dict] |
            TriggerLeg |
            tuple[TriggerLeg] |
            list[TriggerLeg] |
            None
        ),
    ) -> list[TriggerLeg]:
        if legs is None:
            return None

        if isinstance(legs, tuple):
            legs = list(legs)
        elif not isinstance(legs, list):
            legs = [legs]

        _legs = []
        for leg in legs:
            if isinstance(leg, dict):
                leg = TriggerLeg(**leg)
            if not isinstance(leg, TriggerLeg):
                raise TypeError(f"invalid trigger leg: {leg}")
            _legs.append(leg)

        return _legs or None

    @property
    def has_legs(self):
        return bool(self._legs)

    @property
    def n_legs(self):
        return len(self.legs) if self.has_legs else 0

    @property
    def hlt_field(self):
        # remove the first four "HLT_" characters
        return self.name[4:]
