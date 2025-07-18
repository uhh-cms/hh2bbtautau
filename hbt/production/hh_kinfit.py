import functools
from columnflow.production import Producer, producer
from columnflow.util import maybe_import, dev_sandbox
from columnflow.columnar_util import EMPTY_FLOAT, set_ak_column
from columnflow.production.util import attach_coffea_behavior



np = maybe_import("numpy")
ak = maybe_import("awkward")
# maybe_import("hhkinfit2")


set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses=(
        "Electron.*", "Tau.*", "Jet.*", "HHBJet.*", "PuppiMET.*",
        attach_coffea_behavior,
    ),
    produces={
        "HHKinFit.*",
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_kinfit.sh"),
)
def hh_kinfit(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](
        events,
        collections={"HHBJet": {"type_name": "Jet"}},
        **kwargs,
    )
    import hhkinfit2

    n_bjets = ak.num(events.HHBJet, axis=1)
    n_taus = ak.num(events.Tau, axis=1)

    # mask to select events with exactly 2 taus
    ditau_mask = (n_taus == 2)
    diBjet_mask = (n_bjets == 2)
    dihh_mask = ditau_mask & diBjet_mask

    # HHKinFit
    padded_bjets = ak.pad_none(events.HHBJet, 2, clip=True)
    padded_taus = ak.pad_none(events.Tau, 2, clip=True)
    dummy_bjet = {"pt": 0.0, "eta": 0.0, "phi": 0.0, "mass": 1.0}
    dummy_tau = {"pt": 0.0, "eta": 0.0, "phi": 0.0, "mass": 1.0}
    bjets = ak.fill_none(padded_bjets, dummy_bjet)
    taus = ak.fill_none(padded_taus, dummy_tau)
    met = events.PuppiMET
    # change the bjet resolution how you wish, make sure you give it as a double
    bjet_resolutions = [-1.0]
    # bjet_resolutions = [-1.0, 0.0]

    Bjet1, Bjet2 = ak.unzip(ak.combinations(bjets, 2, axis=1))
    tau1, tau2 = taus[:, 0], taus[:, 1]

    b1_list = ak.flatten([list(x) for x in ak.to_list(ak.zip([Bjet1["pt"], Bjet1["eta"], Bjet1["phi"], Bjet1["mass"]], depth_limit=1))], axis=-1)
    b2_list = ak.flatten([list(x) for x in ak.to_list(ak.zip([Bjet2["pt"], Bjet2["eta"], Bjet2["phi"], Bjet2["mass"]], depth_limit=1))], axis=-1)

    tau1_list = ak.Array([list(x) for x in ak.to_list(ak.zip([tau1["pt"], tau1["eta"], tau1["phi"], tau1["mass"]]))])
    tau2_list = ak.Array([list(x) for x in ak.to_list(ak.zip([tau2["pt"], tau2["eta"], tau2["phi"], tau2["mass"]]))])

    met_list= ak.Array([list(x) for x in ak.to_list(ak.zip([met["pt"], met["phi"]]))])
    cov_met_list = ak.Array([list(x) for x in ak.to_list(ak.zip([met["covXX"], met["covXY"], met["covYY"]]))])
    
    print("objects are set as lists")

    results_cpp = hhkinfit2.batchFit(b1_list, b2_list, tau1_list, tau2_list, met_list, cov_met_list, bjet_resolutions)
    results = ak.Array(results_cpp)

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
    
    for i, b_res in enumerate(bjet_resolutions):
        offset = i * 6  # 6 results per resolution
        suffix = f"{int(b_res)}" if b_res != -1.0 else ""

        events = save_interesting_properties(events, f"HHKinFit.MH{suffix}", results[:, offset], dihh_mask)
        events = save_interesting_properties(events, f"HHKinFit.tau1ratio{suffix}", results[:, offset + 1], dihh_mask)
        events = save_interesting_properties(events, f"HHKinFit.tau2ratio{suffix}", results[:, offset + 2], dihh_mask)
        events = save_interesting_properties(events, f"HHKinFit.FitProb{suffix}", results[:, offset + 3], dihh_mask)
        events = save_interesting_properties(events, f"HHKinFit.Chi2{suffix}", results[:, offset + 4], dihh_mask)
        events = save_interesting_properties(events, f"HHKinFit.Convergence{suffix}", results[:, offset + 5], dihh_mask)

    return events