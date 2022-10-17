# coding: utf-8

"""
Producers for btag scale factor weights.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={
        "Jet.hadronFlavour", "Jet.eta", "Jet.pt", "Jet.btagDeepFlavB",
    },
    produces={
        "btag_weight",
    },
)
def btag_sf(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    B-tag scale factor weight producer. Requires an external file in the config as (e.g.)

    .. code-block:: python

        "btag_sf_corr": ("/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-d0a522ea/POG/BTV/2017_UL/btagging.json.gz", "v1"),  # noqa

    as well as an auxiliary entry in the config to refer to the b-tag correction set.

    .. code-block:: python

        cfg.x.btag_sf_correction_set = "deepJet_shape"
    """
    from IPython import embed; embed()

    return events


@btag_sf.requires
def btag_sf_requires(self: Producer, reqs: dict) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@btag_sf.setup
def btag_sf_setup(self: Producer, reqs: dict, inputs: dict) -> None:
    bundle = reqs["external_files"]

    # create the btag sf corrector
    import correctionlib
    correction_set = correctionlib.CorrectionSet.from_string(
        bundle.files.btag_sf_corr.load(formatter="gzip").decode("utf-8"),
    )
    self.btag_sf_corrector = correction_set[self.config_inst.x.btag_sf_correction_set]
