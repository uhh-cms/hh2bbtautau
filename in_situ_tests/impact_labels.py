# coding: utf-8
"""
Nuisance label renaming for the inference tools' pulls & impacts plots (--labels <this file>).

rename_nuisance(name) returns the new label, or None to keep the name. Labels may contain latex
commands (\\tau, \\bar, \\kappa, \\rightarrow, \\minus, ...), dhi.util.to_root_latex converts them
to ROOT latex (backslash -> '#').

Some labels depend on what else is in the plot (e.g. the in-situ TF label only carries the channel
when several TF parameters are shown). The plotting code calls this function once per parameter
from dhi.plots.util.make_parameter_label_map, whose local "parameter_names" holds the full list;
_context() reads it from the call stack and falls back to the single-parameter forms otherwise.
"""
import re
import sys

ERA = r"(?:20\d\d[A-Za-z]*)"
CAT = r"(?:[a-z]+_(?:res[12]b|boosted))"
NUM = r"(?:m?\d+(?:p\d+)?)"

CHANNELS = {
    "etau": r"e\tau_{h}",
    "mutau": r"\mu\tau_{h}",
    "tautau": r"\tau_{h}\tau_{h}",
}

TTBAR = "tt"

PROCESSES = {
    "ttbar": TTBAR,
    "ttbarV": TTBAR + "V",
    "ttbarVV": TTBAR + "VV",
    "singlet": "single top",
    "DY": "DY",
    "EWK": "EWK",
    "W": "W+jets",
    "VV": "VV",
    "VVV": "VVV",
    "QCD": "QCD",
}

SINGLE_H = {
    "ggH": "ggF H ({})",
    "qqH": "VBF H ({})",
    "WH": "WH ({})",
    "ZH": "ZH ({})",
    "ttH": TTBAR + "H ({})",
}
# single Higgs decays in parentheses: ZH (bb); for the HH samples the default bbtautau decay is
# not shown and other decays are appended without parentheses: VBF HH bbWW (1, 0, 1)
H_DECAYS = {"hbb": "bb", "htt": r"\tau\tau"}
HH_DECAYS = {"hbbhtt": "HH", "hbbhww": "HH bbWW"}


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def _num(s):
    """'m1p21' -> '\\minus1.21', '2p45' -> '2.45'"""
    s = s.replace("p", ".")
    return (r"\minus" + s[1:]) if s.startswith("m") else s


def _cat_label(cat):
    """'tautau_res1b' -> '\\tau_{h}\\tau_{h} res1b'"""
    ch, sub = cat.split("_", 1)
    return "{} {}".format(CHANNELS.get(ch, ch), sub)


def _hh_label(mode, couplings, decay):
    """
    couplings: list of (symbol, value string from the process name), e.g.
    "ggF HH (2.45, 1)", "VBF HH bbWW (1, 0, 1)" with the couplings in the order of the process name
    """
    decay = HH_DECAYS.get(decay, "HH {}".format(decay))
    values = ", ".join(_num(v) for _, v in couplings)
    return "{} {} ({})".format(mode, decay, values)


def process_label(proc):
    """Label for a datacard process name (with era and decay), the _QCDCR marker is ignored."""
    return _process_label(proc)[0]


def _process_label(proc):
    """Returns (label, is_cr) for a datacard process name."""
    m = re.match(r"^ggHH_kl_(" + NUM + r")_kt_(" + NUM + r")_13p6TeV_" + ERA + r"_(hbbhtt|hbbhww)$", proc)
    if m:
        return _hh_label("ggF", [(r"\kappa_{\lambda}", m.group(1)), (r"\kappa_{t}", m.group(2))], m.group(3)), False

    m = re.match(r"^qqHH_CV_(" + NUM + r")_C2V_(" + NUM + r")_kl_(" + NUM + r")_13p6TeV_" + ERA + r"_(hbbhtt|hbbhww)$", proc)
    if m:
        couplings = [(r"\kappa_{V}", m.group(1)), (r"\kappa_{2V}", m.group(2)), (r"\kappa_{\lambda}", m.group(3))]
        return _hh_label("VBF", couplings, m.group(4)), False

    m = re.match(r"^(ggH|qqH|WH|ZH|ttH)_13p6TeV_" + ERA + r"(_QCDCR)?_(hbb|htt)$", proc)
    if m:
        return SINGLE_H[m.group(1)].format(H_DECAYS[m.group(3)]), bool(m.group(2))

    m = re.match(r"^([A-Za-z]+)_" + ERA + r"(_QCDCR)?$", proc)
    if m:
        return PROCESSES.get(m.group(1), m.group(1)), bool(m.group(2))

    return proc, False


# ---------------------------------------------------------------------------------------------
# plot context (which parameters are shown together)
# ---------------------------------------------------------------------------------------------

_CTX_CACHE = {}


def _context():
    names = ()
    try:
        f = sys._getframe(1)
        while f is not None:
            if "parameter_names" in f.f_locals:
                names = tuple(f.f_locals["parameter_names"])
                break
            f = f.f_back
    except Exception:
        pass

    key = hash(names)
    if key not in _CTX_CACHE:
        eras, cats = set(), set()
        n_tf = n_lumi = 0
        for n in names:
            m = re.match(r"^prop_bin(?:cat_)?(" + ERA + r")_(" + CAT + r")_bin\d+", n)
            if not m:
                m = re.match(r"^CMS_bbtt_QCD_TF_(" + ERA + r")_(" + CAT + r")$", n)
                if m:
                    n_tf += 1
            if m:
                eras.add(m.group(1))
                cats.add(m.group(2))
            if re.match(r"^lumi_13p6TeV_", n):
                n_lumi += 1
        _CTX_CACHE[key] = dict(
            multi_era=len(eras) > 1,
            multi_cat=len(cats) > 1,
            multi_tf=n_tf > 1,
            multi_lumi=n_lumi > 1,
        )
    return _CTX_CACHE[key]


def _location(era, cat, ctx, sep=", "):
    """'2023BPix, \\tau_{h}\\tau_{h} res1b' depending on what is ambiguous in the current plot"""
    parts = []
    if ctx["multi_era"]:
        parts.append(era)
    if ctx["multi_cat"]:
        parts.append(_cat_label(cat))
    return sep.join(parts)


# ---------------------------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------------------------

def rename_nuisance(name):
    ctx = _context()

    # autoMCStats and in-situ data parameters: prop_bincat_<era>_<cat>_bin<N>[_<process>]
    m = re.match(r"^prop_bin(?:cat_)?(" + ERA + r")_(" + CAT + r")_bin(\d+)(?:_(.+))?$", name)
    if m:
        era, cat, b, proc = m.groups()
        loc = _location(era, cat, ctx, sep=" ")
        where = "{}bin {}".format(loc + " " if loc else "", b)
        if proc is None:
            return "MC stat., {}".format(where)
        if re.match(r"^QCD_data_" + ERA + r"_QCDCR$", proc):
            return "Data stat. CR, {}".format(where)
        label, is_cr = _process_label(proc)
        return "MC stat.{}, {}, {}".format(" CR" if is_cr else "", label, where)

    # in-situ transfer factor statistical uncertainty
    m = re.match(r"^CMS_bbtt_QCD_TF_(" + ERA + r")_(" + CAT + r")$", name)
    if m:
        label = "in-situ TF stat."
        if ctx["multi_tf"]:
            loc = _location(m.group(1), m.group(2), dict(ctx, multi_cat=True))
            label += " ({})".format(loc)
        return label

    # luminosity
    m = re.match(r"^lumi_13p6TeV_(20\d\d)$", name)
    if m:
        return "Luminosity ({}, uncorr.)".format(m.group(1))
    m = re.match(r"^lumi_13p6TeV_((?:\d\d_)+\d\d)$", name)
    if m:
        years = m.group(1).split("_")
        return "Luminosity (20{}-{}, corr.)".format(years[0], years[-1])

    # DY correction
    m = re.match(r"^CMS_DY_(syst|stat)_(\d{4})$", name)
    if m:
        return "DY SF, {}.{}".format(m.group(1), " ({})".format(m.group(2)) if ctx["multi_era"] else "")

    # b-tagging
    m = re.match(r"^CMS_btag_(hf|lf)$", name)
    if m:
        return "b-tag SF: {}".format({"hf": "b/c jets", "lf": "udsg jets"}[m.group(1)])
    m = re.match(r"^CMS_btag_(hf|lf)stats(\d)_(\d{4})$", name)
    if m:
        flav = {"hf": "b/c jets", "lf": "udsg jets"}[m.group(1)]
        era = " ({})".format(m.group(3)) if ctx["multi_era"] else ""
        return "b-tag SF: {} stat. {}{}".format(flav, m.group(2), era)
    m = re.match(r"^CMS_btag_cferr(\d)$", name)
    if m:
        return "b-tag SF: c-jet err. {}".format(m.group(1))

    # tau id scale factors (when enabled): CMS_eff_t_tau_<stat1|stat2|syst>_dm<N>[_<era>]
    m = re.match(r"^CMS_eff_t_tau_(stat|syst)(\d?)_dm(\d+)(?:_(" + ERA + r"))?$", name)
    if m:
        kind = "{}.{}".format(m.group(1), " " + m.group(2) if m.group(2) else "")
        era = " ({})".format(m.group(4)) if ctx["multi_era"] and m.group(4) else ""
        return r"\tau_{h} ID SF " + "{}, DM {}{}".format(kind, m.group(3), era)

    fixed = {
        "CMS_top_pT_reweighting": "top-quark p_{T} reweighting",
        "THU_HH": r"\sigma^{ggF}_{HH} (\kappa_{\lambda}-dependent)",
        "QCD_rate_unc": "QCD rate unc. (10%)",
    }
    return fixed.get(name)
