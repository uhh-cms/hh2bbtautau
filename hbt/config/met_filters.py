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
    if config.campaign.x.run == 2:
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

        # remove filters that are not present in 2016
        if config.campaign.x.year == 2016:
            filters.remove("Flag.hfNoisyHitsFilter")
            filters.remove("Flag.ecalBadCalibFilter")

        # same filter for mc and data
        filters = {
            "mc": filters,
            "data": filters,
        }

    elif config.campaign.x.run == 3:
        filters = [
            "Flag.goodVertices",
            "Flag.globalSuperTightHalo2016Filter",
            "Flag.EcalDeadCellTriggerPrimitiveFilter",
            "Flag.BadPFMuonFilter",
            "Flag.BadPFMuonDzFilter",
            "Flag.hfNoisyHitsFilter",
            "Flag.eeBadScFilter",
        ]

        # same filter for mc and data
        filters = {
            "mc": filters,
            "data": filters,
        }

    else:
        assert False

    config.x.met_filters = DotDict.wrap(filters)
