# coding: utf-8

"""
Producers for btag scale factor weights.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, flat_np_view, layout_ak_array

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

    In addition, JEC uncertainty sources are propagated and weight columns are written if an
    auxiliary config entry ``btag_sf_jec_sources`` exists.

    Resources:

       - https://twiki.cern.ch/twiki/bin/view/CMS/BTagShapeCalibration?rev=26
       - https://indico.cern.ch/event/1096988/contributions/4615134/attachments/2346047/4000529/Nov21_btaggingSFjsons.pdf
    """
    # get flat inputs
    flavor = flat_np_view(events.Jet.hadronFlavour, axis=1)
    abs_eta = flat_np_view(abs(events.Jet.eta), axis=1)
    pt = flat_np_view(events.Jet.pt, axis=1)
    b_discr = flat_np_view(events.Jet.btagDeepFlavB, axis=1)

    # determine the position of c-flavored jets
    c_mask = flavor == 4

    # helper to create and store the weight
    def add_weight(syst_name, syst_direction, column_name):
        # start with flat ones and fill scale factors
        sf_flat = np.ones_like(abs_eta)

        # define an assignment mask
        if syst_name in ["hf", "lf", "hfstats1", "hfstats2", "lfstats1", "lfstats2"]:
            mask = ~c_mask
        elif syst_name in ["cferr1", "cferr2"]:
            mask = c_mask
        else:
            mask = Ellipsis

        # get the flat scale factors
        sf_flat[mask] = self.btag_sf_corrector.evaluate(
            syst_name if syst_name == "central" else f"{syst_direction}_{syst_name}",
            flavor[mask],
            abs_eta[mask],
            pt[mask],
            b_discr[mask],
        )

        # enforce the correct shape and create the product via all jets per event
        sf = layout_ak_array(sf_flat.astype(np.float32), events.Jet.pt)
        weight = ak.prod(sf, axis=1, mask_identity=False)

        # save the new column
        return set_ak_column(events, column_name, weight)

    # when the uncertainty is a known jec shift, obtain the propagated effect and do not produce
    # additional systematics
    jec_source = self.shift_inst.x.jec_source if self.shift_inst.has_tag("jec") else None
    btag_sf_jec_source = "" if jec_source == "Total" else jec_source
    if jec_source and btag_sf_jec_source in self.config_inst.x("btag_sf_jec_sources", []):
        # TODO: year dependent jec variations covered?
        events = add_weight(
            f"jes{'' if jec_source == 'Total' else jec_source}",
            self.shift_inst.direction,
            f"btag_weight_jec_{jec_source}_{self.shift_inst.direction}",
        )
    elif self.shift_inst.is_nominal:
        # nominal weight and those of all method intrinsic uncertainties
        events = add_weight("central", None, "btag_weight")
        for syst_name, col_name in self.btag_uncs.items():
            for direction in ["up", "down"]:
                name = col_name.format(year=self.config_inst.campaign.x.year)
                events = add_weight(
                    syst_name,
                    direction,
                    f"btag_weight_{name}_{direction}",
                )
    else:
        # any other shift, just produce the nominal weight
        events = add_weight("central", None, "btag_weight")

    return events


@btag_sf.init
def btag_sf_init(self: Producer) -> None:
    # save names of method-intrinsic uncertainties
    self.btag_uncs = {
        "hf": "hf",
        "lf": "lf",
        "hfstats1": "hfstats1_{year}",
        "hfstats2": "hfstats2_{year}",
        "lfstats1": "lfstats1_{year}",
        "lfstats2": "lfstats2_{year}",
        "cferr1": "cferr1",
        "cferr2": "cferr2",
    }

    # add uncertainty sources of the method itself
    for col_name in self.btag_uncs.values():
        for direction in ["up", "down"]:
            name = col_name.format(year=self.config_inst.campaign.x.year)
            self.produces.add(f"btag_weight_{name}_{direction}")


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
