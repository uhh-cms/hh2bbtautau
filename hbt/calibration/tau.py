# coding: utf-8

"""
Tau energy correction methods.
"""

import functools
import itertools

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.util import propagate_met
from columnflow.util import maybe_import, InsertableDict
from columnflow.columnar_util import set_ak_column, flat_np_view, ak_copy


ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@calibrator(
    uses={
        # nano columns
        "nTau", "Tau.pt", "Tau.eta", "Tau.phi", "Tau.mass", "Tau.charge", "Tau.genPartFlav",
        "Tau.decayMode", "MET.pt", "MET.phi",
    },
    produces={
        "Tau.pt", "Tau.mass", "MET.pt", "MET.phi",
    } | {
        f"{field}_tec_{match}_dm{dm}_{direction}"
        for field, match, dm, direction in itertools.product(
            ["Tau.pt", "Tau.mass", "MET.pt", "MET.phi"],
            ["jet", "e"],
            [0, 1, 10, 11],
            ["up", "down"],
        )
    },
    # only run on mc
    mc_only=True,
    # function to determine the correction file
    get_tau_file=(lambda self, external_files: external_files.tau_sf),
)
def tec(
    self: Calibrator,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Calibrator for tau energy. Requires an external file in the config under ``tau_sf``:

    .. code-block:: python

        cfg.x.external_files = DotDict.wrap({
            "tau_sf": "/afs/cern.ch/work/m/mrieger/public/mirrors/jsonpog-integration-9ea86c4c/POG/TAU/2017_UL/tau.json.gz",  # noqa
        })

    *get_tau_file* can be adapted in a subclass in case it is stored differently in the external
    files. A correction set named ``"tau_energy_scale"`` is extracted from it.

    Resources:
    https://twiki.cern.ch/twiki/bin/view/CMS/TauIDRecommendationForRun2?rev=113
    https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/849c6a6efef907f4033715d52290d1a661b7e8f9/POG/TAU
    """
    # fail when running on data
    if self.dataset_inst.is_data:
        raise ValueError("attempt to apply tau energy corrections in data")

    # the correction tool only supports flat arrays, so convert inputs to flat np view first
    pt = flat_np_view(events.Tau.pt, axis=1)
    mass = flat_np_view(events.Tau.mass, axis=1)
    eta = flat_np_view(events.Tau.eta, axis=1)
    dm = flat_np_view(events.Tau.decayMode, axis=1)
    match = flat_np_view(events.Tau.genPartFlav, axis=1)

    # get the scale factors for the four supported decay modes
    dm_mask = (dm == 0) | (dm == 1) | (dm == 10) | (dm == 11)
    scales_nom = np.ones_like(dm_mask, dtype=np.float32)
    scales_up = np.ones_like(dm_mask, dtype=np.float32)
    scales_down = np.ones_like(dm_mask, dtype=np.float32)

    args = (pt[dm_mask], eta[dm_mask], dm[dm_mask], match[dm_mask], self.config_inst.x.tau_tagger)
    scales_nom[dm_mask] = self.tec_corrector(*args, "nom")
    scales_up[dm_mask] = self.tec_corrector(*args, "up")
    scales_down[dm_mask] = self.tec_corrector(*args, "down")

    # custom adjustment 1: reset where the matching value is unhandled
    # custom adjustment 2: reset electrons faking taus where the pt is too small
    mask1 = (match < 1) | (match > 5)
    mask2 = ((match == 1) | (match == 3) & (pt <= 20.0))

    # apply reset masks
    mask = mask1 | mask2
    scales_nom[mask] = 1.0
    scales_up[mask] = 1.0
    scales_down[mask] = 1.0

    # create varied collections per decay mode
    for (match_mask, match_name), _dm, (direction, scales) in itertools.product(
        [(match == 5, "jet"), ((match == 1) | (match == 3), "e")],
        [0, 1, 10, 11],
        [("up", scales_up), ("down", scales_down)],
    ):
        # copy pt and mass
        pt_varied = ak_copy(events.Tau.pt)
        mass_varied = ak_copy(events.Tau.mass)
        pt_view = flat_np_view(pt_varied, axis=1)
        mass_view = flat_np_view(mass_varied, axis=1)

        # correct pt and mass for taus with that gen match and decay mode
        mask = match_mask & (dm == _dm)
        pt_view[mask] *= scales[mask]
        mass_view[mask] *= scales[mask]

        # propagate changes to MET
        met_pt_varied, met_phi_varied = propagate_met(
            events.Tau.pt,
            events.Tau.phi,
            pt_varied,
            events.Tau.phi,
            events.MET.pt,
            events.MET.phi,
        )

        # save columns
        postfix = f"tec_{match_name}_dm{_dm}_{direction}"
        events = set_ak_column_f32(events, f"Tau.pt_{postfix}", pt_varied)
        events = set_ak_column_f32(events, f"Tau.mass_{postfix}", mass_varied)
        events = set_ak_column_f32(events, f"MET.pt_{postfix}", met_pt_varied)
        events = set_ak_column_f32(events, f"MET.phi_{postfix}", met_phi_varied)

    # apply the nominal correction
    # note: changes are applied to the views and directly propagate to the original ak arrays
    # and do not need to be inserted into the events chunk again
    tau_sum_before = events.Tau.sum(axis=1)
    pt *= scales_nom
    mass *= scales_nom

    # propagate changes to MET
    met_pt, met_phi = propagate_met(
        tau_sum_before.pt,
        tau_sum_before.phi,
        events.Tau.pt,
        events.Tau.phi,
        events.MET.pt,
        events.MET.phi,
    )

    # save columns
    events = set_ak_column_f32(events, "MET.pt", met_pt)
    events = set_ak_column_f32(events, "MET.phi", met_phi)

    return events


@tec.requires
def tec_requires(self: Calibrator, reqs: dict) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@tec.setup
def tec_setup(self: Calibrator, reqs: dict, inputs: dict, reader_targets: InsertableDict) -> None:
    bundle = reqs["external_files"]

    # create the tec corrector
    import correctionlib
    correctionlib.highlevel.Correction.__call__ = correctionlib.highlevel.Correction.evaluate
    correction_set = correctionlib.CorrectionSet.from_string(
        self.get_tau_file(bundle.files).load(formatter="gzip").decode("utf-8"),
    )
    self.tec_corrector = correction_set["tau_energy_scale"]

    # check versions
    assert self.tec_corrector.version in [0]
