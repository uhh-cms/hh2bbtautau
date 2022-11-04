# coding: utf-8

"""
Definition of MET filter flags.
"""

import order as od

from columnflow.util import DotDict


def add_met_filters(config: od.Config) -> None:
    """
    Adds all MET filters to a *config*.

    Resources:
    https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2?rev=157#UL_data
    """
    if config.campaign.x.year == 2016:
        filters = [
            "Flag.goodVertices",
            "Flag.globalSuperTightHalo2016Filter",
            "Flag.HBHENoiseFilter",
            "Flag.HBHENoiseIsoFilter",
            "Flag.EcalDeadCellTriggerPrimitiveFilter",
            "Flag.BadPFMuonFilter",
            "Flag.BadPFMuonDzFilter",
            "Flag.eeBadScFilter",
        ]
        # same filter for mc and data, but still separate
        filters = {
            "mc": filters,
            "data": filters,
        }
    else:
        filters = [
            "Flag.goodVertices",
            "Flag.globalSuperTightHalo2016Filter",
            "Flag.HBHENoiseFilter",
            "Flag.HBHENoiseIsoFilter",
            "Flag.EcalDeadCellTriggerPrimitiveFilter",
            "Flag.BadPFMuonFilter",
            "Flag.BadPFMuonDzFilter",
            "Flag.hfNoisyHitsFilter",
            "Flag.eeBadScFilter",
            "Flag.ecalBadCalibFilter",
        ]
        # same filter for mc and data, but still separate
        filters = {
            "mc": filters,
            "data": filters,
        }

    config.x.met_filters = DotDict.wrap(filters)
