# coding: utf-8

"""
Definition of triggers.

General requirement from the lepton selection:
For cross triggers, the lepton leg (lepton= {"e", "mu"}) must be defined before the tau leg.
An error here would be caught in the lepton selection, but it is better to avoid it.

Convention for Ids:
- 1xx: single muon triggers
- 2xx: single electron triggers
- 3xx: mu-tau triggers
- 4xx: e-tau triggers
- 5xx: tau-tau triggers
- 6xx: vbf triggers
- 7xx: tau tau jet triggers
- 8xx: quadjet triggers

Starting from xx = 01 and with a unique name for each path across all years.

Current status:
101 -> HLT_IsoMu22
102 -> HLT_IsoMu22_eta2p1
103 -> HLT_IsoTkMu22
104 -> HLT_IsoTkMu22_eta2p1
105 -> HLT_IsoMu24
106 -> HLT_IsoMu27

201 -> HLT_Ele25_eta2p1_WPTight_Gsf
202 -> HLT_Ele32_WPTight_Gsf
203 -> HLT_Ele32_WPTight_Gsf_L1DoubleEG
204 -> HLT_Ele35_WPTight_Gsf
205 -> HLT_Ele30_WPTight_Gsf

301 -> HLT_IsoMu19_eta2p1_LooseIsoPFTau20
302 -> HLT_IsoMu19_eta2p1_LooseIsoPFTau20_SingleL1
303 -> HLT_IsoMu20_eta2p1_LooseChargedIsoPFTau27_eta2p1_CrossL1
304 -> HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1

401 -> HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_SingleL1
402 -> HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau20
403 -> HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau30
404 -> HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTau30_eta2p1_CrossL1
405 -> HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1

501 -> HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Reg
502 -> HLT_DoubleMediumCombinedIsoPFTau35_Trk1_eta2p1_Reg
503 -> HLT_DoubleMediumChargedIsoPFTau35_Trk1_eta2p1_Reg
504 -> HLT_DoubleTightChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg
505 -> HLT_DoubleMediumChargedIsoPFTau40_Trk1_TightID_eta2p1_Reg
506 -> HLT_DoubleTightChargedIsoPFTau40_Trk1_eta2p1_Reg
507 -> HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1
508 -> HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1
509 -> HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1

601 -> HLT_VBF_DoubleLooseChargedIsoPFTau20_Trk1_eta2p1_Reg
602 -> HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1
603 -> HLT_VBF_DoubleLooseChargedIsoPFTauHPS20_Trk1_eta2p1
604 -> HLT_DoublePFJets40_Mass500_MediumDeepTauPFTauHPS45_L2NN_MediumDeepTauPFTauHPS20_eta2p1
605 -> HLT_VBF_DiPFJet105_40_Mjj1000_Detajj3p5
606 -> HLT_VBF_DiPFJet90_40_Mjj600_Detajj2p5_Mu3_TrkIsoVVL
607 -> HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Ele12_eta2p1_WPTight_Gsf
608 -> HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Ele17_eta2p1_WPTight_Gsf
609 -> HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_MediumDeepTauPFTauHPS45_L2NN_eta2p1

701 -> HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60
702 -> HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet75
"""

from __future__ import annotations

import functools

import order as od

from columnflow.util import DotDict

from hbt.config.util import Trigger, TriggerLeg, TriggerBits as Bits


# use the CCLub names for the trigger bits and improve them when necessary
trigger_bits = DotDict.wrap({
    # for v12:
    # checked with https://github.com/cms-sw/cmssw/blob/CMSSW_13_0_X/PhysicsTools/NanoAOD/python/triggerObjects_cff.py
    # and in particular https://github.com/cms-sw/cmssw/blob/2defd844e96613d2438b690d10b79c773e02ab57/PhysicsTools/NanoAOD/python/triggerObjects_cff.py  # noqa: E501
    # for v14:
    # from https://github.com/cms-sw/cmssw/tree/f50cf84669608dbe67fd8430660abe651d5b46fd/PhysicsTools/NanoAOD/python/triggerObjects_cff.py  # noqa: E501
    # last update in https://github.com/cms-sw/cmssw/blob/CMSSW_14_0_X/PhysicsTools/NanoAOD/python/triggerObjects_cff.py
    # for v15:
    # checked with https://github.com/cms-sw/cmssw/blob/CMSSW_15_0_X/PhysicsTools/NanoAOD/python/triggerObjects_cff.py#L235C1-L237C152  # noqa: E501
    # and in particular https://github.com/cms-sw/cmssw/blob/66b501ce9c2458f70674d1b977771ad22fe448eb/PhysicsTools/NanoAOD/python/triggerObjects_cff.py  # noqa: E501


    "e": {
        "CaloIdLTrackIdLIsoVL": Bits(v12=1, v14="v12", v15="v12"),
        "WPTightTrackIso": Bits(v12=2, v14="v12", v15="v12"),
        "WPLooseTrackIso": Bits(v12=4, v14="v12", v15="v12"),
        "OverlapFilterPFTau": Bits(v12=8, v14="v12", v15="v12"),
        "DiElectron": Bits(v12=16),
        "DiElectronLeg1": Bits(v14=16, v15="v14"),
        "DiElectronLeg2": Bits(v14=32, v15="v14"),
        "MuEle": Bits(v12=32, v14=64, v15="v14"),
        "EleTau": Bits(v12=64, v14=128, v15="v14"),
        "TripleElectron": Bits(v12=128, v14=256, v15="v14"),
        "SingleMuonDiEle": Bits(v12=256, v14=512, v15="v14"),
        "DiMuonSingleEle": Bits(v12=512, v14=1024, v15="v14"),
        "SingleEle_L1DoubleAndSingleEle": Bits(v12=1024, v14=2048, v15="v14"),
        "SingleEle_CaloIdVT_GsfTrkIdT": Bits(v12=2048, v14=4096, v15="v14"),
        "SingleEle_PFJet": Bits(v12=4096, v14=8192, v15="v14"),
        "Photon175_Photon200": Bits(v12=8192, v14=16384, v15="v14"),
        "DoubleEle_CaloIdL_MW_seeded": Bits(v14=32768, v15="v14"),
        "DoubleEle_CaloIdL_MW_unseeded": Bits(v14=65536, v15="v14"),
        "EleTauPNet": Bits(v14=131072, v15="v14"),
        "SingleEle_HLT30WPTightGSfTrackIso": Bits(v15=262144),
        "VBFWPTightGsfTrackIso": Bits(v15=524288),
    },
    "mu": {
        "TrkIsoVVL": Bits(v12=1, v14="v12", v15="v12"),
        "Iso": Bits(v12=2, v14="v12", v15="v12"),
        "OverlapFilterPFTau": Bits(v12=4, v14="v12", v15="v12"),
        "SingleMuon": Bits(v12=8, v14="v12", v15="v12"),
        "DiMuon": Bits(v12=16, v14="v12", v15="v12"),
        "MuEle": Bits(v12=32, v14="v12", v15="v12"),
        "MuTau": Bits(v12=64, v14="v12", v15="v12"),
        "TripleMuon": Bits(v12=128, v14="v12", v15="v12"),
        "DiMuonSingleEle": Bits(v12=256, v14="v12", v15="v12"),
        "SingleMuonDiEle": Bits(v12=512, v14="v12", v15="v12"),
        "Mu50": Bits(v12=1024, v14="v12", v15="v12"),
        "Mu100": Bits(v12=2048, v14="v12", v15="v12"),
        "SingleMuonSinglePhoton": Bits(v12=4096, v14="v12", v15="v12"),
        "MuTauPNet": Bits(v14=8192, v15="v14"),
    },
    "tau": {  # general comment: lot of v14 paths contain PNet paths, not available in v12, e.g. OverlapFilterIsoEle
        "LooseChargedIso": Bits(v12=1),
        "Loose": Bits(v14=1, v15="v14"),
        "MediumChargedIso": Bits(v12=2),
        "Medium": Bits(v14=2, v15="v14"),
        "TightChargedIso": Bits(v12=4),
        "Tight": Bits(v14=4, v15="v14"),
        "DeepTau": Bits(v12=8, v14="v12", v15="v12"),
        "PNet": Bits(v14=16, v15="v14"),
        "TightOOSCPhotons": Bits(v12=16),
        "HPS": Bits(v12=32, v14=268435456, v15="v14"),
        "ChargedIso": Bits(v14=32, v15="v14"),
        "ChargedIsoDiTau": Bits(v12=64),
        "Dxy": Bits(v14=64, v15="v14"),
        "DeepTauDiTau": Bits(v12=128, v14=2048 + 8, v15="v14"),  # manually created bit combinations for v14,v15
        "ETauFilter": Bits(v14=128, v15="v14"),
        "MuTauFilter": Bits(v14=256, v15="v14"),
        "OverlapFilterIsoEle": Bits(v12=256, v14=4096, v15="v14"),  # contains HPS in v14, not in v12
        "OverlapFilterIsoMu": Bits(v12=512, v14=8192, v15="v14"),  # contains HPS in v14, not in v12
        "SingleTau": Bits(v14=512, v15="v14"),
        "SingleTauOrTauMet": Bits(v12=1024),  # more general paths than SingleTau in v14
        "VBFDiTau": Bits(v14=1024, v15="v14"),
        "VBFpDoublePFTau_run2": Bits(v12=2048),
        "VBFpDoublePFTau_run3": Bits(v12=4096),  # warning: this trigger bit expects "ChargedIso" in the filter name, this does not correspond to our actual VBF filter name  # noqa: E501
        "DiTau": Bits(v14=2048, v15="v14"),
        "DiPFJetAndDiTau": Bits(v12=8192),
        "DiTauAndPFJet": Bits(v12=16384, v14="v12", v15="v12"),
        "DisplacedTau": Bits(v12=32768),
        "ETauDisplaced": Bits(v14=32768, v15="v14"),
        "MuTauDisplaced": Bits(v14=65536, v15="v14"),
        "DiTauDisplaced": Bits(v14=131072, v15="v14"),
        "Monitoring": Bits(v12=65536, v14=262144, v15="v14"),
        "MonitoringForVBFIsoTau": Bits(v14=524288, v15="v14"),
        "MonitoringDiTauAndPFJet": Bits(v14=1048576, v15="v14"),
        "MonitoringMuTauDisplaced": Bits(v14=2097152, v15="v14"),
        "MonitoringDiTau": Bits(v14=8388608, v15="v14"),
        "VBFDoubleTauMonitoring": Bits(v14=33554432, v15="v14"),
        "SingleTauMonitoring": Bits(v14=67108864, v15="v14"),
        "OverlapFilter": Bits(v14=16777216, v15="v14"),
        "RegionalPaths": Bits(v12=131072),
        "L1SeededPaths": Bits(v12=262144),
        "MatchL1HLT": Bits(v12=262144, v14=134217728, v15="v14"),  # for v12: alias for v12-v14 compatibility
        "1Prong": Bits(v12=524288),
        "OneProng": Bits(v14=4194304, v15="v14"),  # just changed "1" to "One" for v14, still means different filters
        "SinglePFTauFilter": Bits(v14=536870912, v15="v14"),
        "VBFSingleTau": Bits(v14=1073741824, v15="v14"),
    },
    "jet": {
        "4PixelOnlyPFCentralJetTightIDPt20": Bits(v12=1, v14="v12", v15="v12"),
        "3PixelOnlyPFCentralJetTightIDPt30": Bits(v12=2, v14="v12", v15="v12"),
        "PFJetFilterTwoC30": Bits(v12=4, v14="v12", v15="v12"),
        "4PFCentralJetTightIDPt30": Bits(v12=8, v14="v12", v15="v12"),
        "4PFCentralJetTightIDPt35": Bits(v12=16, v14="v12", v15="v12"),
        "QuadCentralJet30": Bits(v12=32, v14="v12", v15="v12"),
        "2PixelOnlyPFCentralJetTightIDPt40": Bits(v12=64, v14="v12", v15="v12"),
        "L1sTripleJetVBF_orHTT_orDoubleJet_orSingleJet": Bits(v12=128, v14="v12", v15="v12"),
        "3PFCentralJetTightIDPt40": Bits(v12=256, v14="v12", v15="v12"),
        "3PFCentralJetTightIDPt45": Bits(v12=512, v14="v12", v15="v12"),
        "L1sQuadJetsHT": Bits(v12=1024, v14="v12", v15="v12"),
        "BTagCaloDeepCSVp17Double": Bits(v12=2048, v14="v12", v15="v12"),
        "PFCentralJetLooseIDQuad30": Bits(v12=4096, v14="v12", v15="v12"),
        "1PFCentralJetLooseID75": Bits(v12=8192, v14="v12", v15="v12"),
        "2PFCentralJetLooseID60": Bits(v12=16384, v14="v12", v15="v12"),
        "3PFCentralJetLooseID45": Bits(v12=32768, v14="v12", v15="v12"),
        "4PFCentralJetLooseID40": Bits(v12=65536, v14="v12", v15="v12"),
        "DoubleTau+Jet": Bits(v12=131072, v14="v12", v15="v12"),  # v14 also contains PNet paths
        "VBFcrossCleanedDeepTauPFTau": Bits(v12=262144, v14="v12", v15="v12"),  # more general VBFDiTauJets in v14  TODO: change name?  # noqa: E501
        "VBFcrossCleanedUsingDijetCorr": Bits(v12=524288, v14="v12", v15="v12"),  # more general VBFSingleTauJets in v14  TODO: change name?  # noqa: E501
        "MuonTauJetMonitoring": Bits(v12=1048576, v14="v12", v15="v12"),
        "2PFCentralJetTightIDPt50": Bits(v12=2097152, v14="v12", v15="v12"),
        "1PixelOnlyPFCentralJetTightIDPt60": Bits(v12=4194304, v14="v12", v15="v12"),
        "1PFCentralJetTightIDPt70": Bits(v12=8388608, v14="v12", v15="v12"),
        "BTagPFDeepJet1p5Single": Bits(v12=16777216, v14="v12", v15="v12"),
        "BTagPFDeepJet4p5Triple": Bits(v12=33554432, v14="v12", v15="v12"),
        "PFJetTwoC30PFBTagParticleNet2BTagSum0p65ORPFCentralJetPt30PNet2BTagMean0p55": Bits(v12=67108864, v14="v12", v15="v12"),  # noqa: E501
        "2PixelOnlyPFCentralJetTightIDPt20OR1PixelOnlyPFCentralJetTightIDPt50": Bits(v12=134217728, v14="v12", v15="v12"),  # noqa: E501
        "2PFCentralJetTightIDPt30": Bits(v12=268435456, v14="v12", v15="v12"),
        "1PFCentralJetTightIDPt60": Bits(v12=536870912, v14="v12", v15="v12"),
        "PF2CentralJetPt30PNet2BTagMean0p50": Bits(v12=1073741824, v14="v12", v15="v12"),
        "4PFCentralJetPt25": Bits(v15=2147483648),
        "PFCentralJetNoIDPt25PNet1BTag0p20": Bits(v15=4294967296),
        "PFCentralJetNoIDPt25PNet1TauHTag0p50": Bits(v15=8589934592),
        "4PFCentralJetTightIDPt25": Bits(v15=17179869184),
        "PFCentralJetPt25PNet2BTagMean0p55": Bits(v15=34359738368),
        "VBFincl": Bits(v15=68719476736),
        "VBF+dijet": Bits(v15=137438953472),
        "VBFmet": Bits(v15=274877906944),
        "VBFmu": Bits(v15=549755813888),
        "VBFgamma": Bits(v15=1099511627776),
        "VBFele": Bits(v15=2199023255552),
        "SinglePFJetX": Bits(v15=4398046511104),
        "SinglePFJetFwdX": Bits(v15=8796093022208),
        "DiPFJetAveX": Bits(v15=17592186044416),
        "DiPFJetAveX_HFJEC": Bits(v15=35184372088832),
    },
})


def get_bit_sum(nano_version: int, obj_name: str, names: list[str | None]) -> int:
    return sum(
        trigger_bits[obj_name][name].get(nano_version)
        for name in names
        if name is not None
    ) or None


# 2016 triggers as per AN of CMS-HIG-20-010 (AN2018_121_v11-1)
def add_triggers_2016(config: od.Config) -> None:
    """
    Adds all triggers to a *config*. For the conversion from filter names to trigger bits, see
    https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py.
    """
    config.x.triggers = od.UniqueObjectIndex(Trigger)

    #
    # e tauh
    #
    # from https://twiki.cern.ch/twiki/bin/view/CMS/TauTrigger#Tau_Triggers_in_NanoAOD_2016
    config.x.triggers.add(
        name="HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_SingleL1",
        id=401,
        legs=dict(
            e=TriggerLeg(
                pdg_id=11,
                # min_pt=26.0,  # TODO
                # filter names:
                #
                trigger_bits=None,  # TODO
            ),
            tau=TriggerLeg(
                pdg_id=15,
                # min_pt=22.0,  # TODO
                # filter names:
                #
                trigger_bits=None,  # TODO
            ),
        ),
        # does not exist for run F on but should only be used until run 276215 -> which era?
        # TODO: to be checked
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era <= "E"),
        tags={"cross_trigger", "cross_e_tau"},
    )
    config.x.triggers.add(
        name="HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau20",
        id=402,
        legs=dict(
            e=TriggerLeg(
                pdg_id=11,
                # min_pt=26.0,  # TODO
                # filter names:
                #
                trigger_bits=None,  # TODO
            ),
            tau=TriggerLeg(
                pdg_id=15,
                # min_pt=22.0,  # TODO
                # filter names:
                #
                trigger_bits=None,  # TODO
            ),
        ),
        # does not exist for run F on but should only be used between run 276215 and 278270 -> which eras?
        # TODO: to be checked
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data and dataset_inst.x.era <= "E"),
        tags={"cross_trigger", "cross_e_tau"},
    )
    config.x.triggers.add(
        name="HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau30",
        id=403,
        legs=dict(
            e=TriggerLeg(
                pdg_id=11,
                # min_pt=26.0,  # TODO
                # filter names:
                #
                trigger_bits=None,  # TODO
            ),
            tau=TriggerLeg(
                pdg_id=15,
                # min_pt=32.0,  # TODO
                # filter names:
                #
                trigger_bits=None,  # TODO
            ),
        ),
        # does not exist until run E but should only be used after run 278270 -> which era?
        # TODO: to be checked
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data and dataset_inst.x.era >= "E"),
        tags={"cross_trigger", "cross_e_tau"},
    )

    #
    # mu tauh
    #
    config.x.triggers.add(
        name="HLT_IsoMu19_eta2p1_LooseIsoPFTau20",
        id=301,
        legs=dict(
            mu=TriggerLeg(
                pdg_id=13,
                # min_pt=22,  # TODO
                # filter names:
                # TODO
                trigger_bits=None,  # TODO
            ),
            tau=TriggerLeg(
                pdg_id=15,
                # min_pt=23,  # TODO
                # filter names:
                # TODO
                trigger_bits=None,  # TODO
            ),
        ),
        tags={"cross_trigger", "cross_mu_tau"},
    )
    config.x.triggers.add(
        name="HLT_IsoMu19_eta2p1_LooseIsoPFTau20_SingleL1",
        id=302,
        legs=dict(
            mu=TriggerLeg(
                pdg_id=13,
                # min_pt=22,  # TODO
                # filter names:
                # TODO
                trigger_bits=None,  # TODO
            ),
            tau=TriggerLeg(
                pdg_id=15,
                # min_pt=23,  # TODO
                # filter names:
                # TODO
                trigger_bits=None,  # TODO
            ),
        ),
        tags={"cross_trigger", "cross_mu_tau"},
    )

    #
    # tauh tauh
    #
    config.x.triggers.add(
        name="HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Reg",
        id=501,
        legs=dict(
            tau1=TriggerLeg(
                pdg_id=15,
                # min_pt=38,  # TODO
                # filter names:
                # TODO
                trigger_bits=None,  # TODO
            ),
            tau2=TriggerLeg(
                pdg_id=15,
                # min_pt=38,  # TODO
                # filter names:
                # TODO
                trigger_bits=None,  # TODO
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or ("B" <= dataset_inst.x.era <= "F")),
        tags={"cross_trigger", "cross_tau_tau"},
    )
    config.x.triggers.add(
        name="HLT_DoubleMediumCombinedIsoPFTau35_Trk1_eta2p1_Reg",
        id=502,
        legs=dict(
            tau1=TriggerLeg(
                pdg_id=15,
                # min_pt=38,  # TODO
                # filter names:
                # TODO
                trigger_bits=None,  # TODO
            ),
            tau2=TriggerLeg(
                pdg_id=15,
                # min_pt=38,  # TODO
                # filter names:
                # TODO
                trigger_bits=None,  # TODO
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "H"),
        tags={"cross_trigger", "cross_tau_tau"},
    )

    #
    # vbf
    #
    # none

    if config.campaign.has_tag("preVFP"):
        #
        # single electron
        #
        config.x.triggers.add(
            name="HLT_Ele25_eta2p1_WPTight_Gsf",
            id=201,
            legs=dict(
                e=TriggerLeg(
                    pdg_id=11,
                    # min_pt=28,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
            ),
            tags={"single_trigger", "single_e"},
        )

        #
        # single muon
        #
        config.x.triggers.add(
            name="HLT_IsoMu22",
            id=101,
            legs=dict(
                mu=TriggerLeg(
                    pdg_id=13,
                    # min_pt=25,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
            ),
            tags={"single_trigger", "single_mu"},
        )
        config.x.triggers.add(
            name="HLT_IsoMu22_eta2p1",
            id=102,
            legs=dict(
                mu=TriggerLeg(
                    pdg_id=13,
                    # min_pt=25,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
            ),
            tags={"single_trigger", "single_mu"},
        )
        config.x.triggers.add(
            name="HLT_IsoTkMu22",
            id=103,
            legs=dict(
                mu=TriggerLeg(
                    pdg_id=13,
                    # min_pt=25,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
            ),
            tags={"single_trigger", "single_mu"},
        )
        config.x.triggers.add(
            name="HLT_IsoTkMu22_eta2p1",
            id=104,
            legs=dict(
                mu=TriggerLeg(
                    pdg_id=13,
                    # min_pt=25,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
            ),
            tags={"single_trigger", "single_mu"},
        )


def add_triggers_2017(config: od.Config) -> None:
    """
    Adds all triggers to a *config*. For the conversion from filter names to trigger bits, see
    https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py.
    """
    config.x.triggers = od.UniqueObjectIndex(Trigger)

    #
    # single electron
    #
    config.x.triggers.add(
        name="HLT_Ele32_WPTight_Gsf",
        id=202,
        legs=dict(
            e=TriggerLeg(
                pdg_id=11,
                # min_pt=35.0,
                # filter names:
                # hltEle32WPTightGsfTrackIsoFilter
                trigger_bits=2,
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "D"),
        tags={"single_trigger", "single_e"},
    )
    config.x.triggers.add(
        name="HLT_Ele32_WPTight_Gsf_L1DoubleEG",
        id=203,
        legs=dict(
            e=TriggerLeg(
                pdg_id=11,
                # min_pt=35.0,
                # filter names:
                # hltEle32L1DoubleEGWPTightGsfTrackIsoFilter
                # hltEGL1SingleEGOrFilter
                trigger_bits=2 + 1024,
            ),
        ),
        tags={"single_trigger", "single_e"},
    )
    config.x.triggers.add(
        name="HLT_Ele35_WPTight_Gsf",
        id=204,
        legs=dict(
            e=TriggerLeg(
                pdg_id=11,
                # min_pt=38.0,
                # filter names:
                # hltEle35noerWPTightGsfTrackIsoFilter
                trigger_bits=2,
            ),
        ),
        tags={"single_trigger", "single_e"},
    )

    #
    # single muon
    #
    config.x.triggers.add(
        name="HLT_IsoMu24",
        id=105,
        legs=dict(
            mu=TriggerLeg(
                pdg_id=13,
                # min_pt=26.0,
                # filter names:
                # hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p07
                trigger_bits=2,
            ),
        ),
        tags={"single_trigger", "single_mu"},
    )
    config.x.triggers.add(
        name="HLT_IsoMu27",
        id=106,
        legs=dict(
            mu=TriggerLeg(
                pdg_id=13,
                # min_pt=29.0,
                # filter names:
                # hltL3crIsoL1sMu22Or25L1f0L2f10QL3f27QL3trkIsoFiltered0p07
                trigger_bits=2,
            ),
        ),
        tags={"single_trigger", "single_mu"},
    )

    #
    # e tauh
    #
    config.x.triggers.add(
        name="HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTau30_eta2p1_CrossL1",
        id=404,
        legs=dict(
            e=TriggerLeg(
                pdg_id=11,
                # min_pt=27.0,
                # filter names:
                # hltEle24erWPTightGsfTrackIsoFilterForTau
                # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
                trigger_bits=2 + 64,
            ),
            tau=TriggerLeg(
                pdg_id=15,
                # min_pt=35.0,
                # filter names:
                # hltSelectedPFTau30LooseChargedIsolationL1HLTMatched
                # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
                trigger_bits=1024 + 256,
            ),
        ),
        tags={"cross_trigger", "cross_e_tau"},
    )

    #
    # mu tauh
    #
    config.x.triggers.add(
        name="HLT_IsoMu20_eta2p1_LooseChargedIsoPFTau27_eta2p1_CrossL1",
        id=303,
        legs=dict(
            mu=TriggerLeg(
                pdg_id=13,
                # min_pt=22.0,
                # filter names:
                # hltL3crIsoL1sMu18erTau24erIorMu20erTau24erL1f0L2f10QL3f20QL3trkIsoFiltered0p07
                # hltOverlapFilterIsoMu20LooseChargedIsoPFTau27L1Seeded
                trigger_bits=2 + 64,
            ),
            tau=TriggerLeg(
                pdg_id=15,
                # min_pt=32.0,
                # filter names:
                # hltSelectedPFTau27LooseChargedIsolationAgainstMuonL1HLTMatched or
                # hltOverlapFilterIsoMu20LooseChargedIsoPFTau27L1Seeded
                trigger_bits=1024 + 512,
            ),
        ),
        tags={"cross_trigger", "cross_mu_tau"},
    )

    #
    # tauh tauh
    #
    config.x.triggers.add(
        name="HLT_DoubleMediumChargedIsoPFTau35_Trk1_eta2p1_Reg",
        id=503,
        legs=dict(
            tau1=TriggerLeg(
                pdg_id=15,
                # min_pt=40.0,
                # filter names:
                # hltDoublePFTau35TrackPt1MediumChargedIsolationAndTightOOSCPhotonsDz02Reg
                trigger_bits=64,
            ),
            tau2=TriggerLeg(
                pdg_id=15,
                # min_pt=40.0,
                # filter names:
                # hltDoublePFTau35TrackPt1MediumChargedIsolationAndTightOOSCPhotonsDz02Reg
                trigger_bits=64,
            ),
        ),
        tags={"cross_trigger", "cross_tau_tau"},
    )
    config.x.triggers.add(
        name="HLT_DoubleTightChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg",
        id=504,
        legs=dict(
            tau1=TriggerLeg(
                pdg_id=15,
                # min_pt=40.0,
                # filter names:
                # hltDoublePFTau35TrackPt1TightChargedIsolationAndTightOOSCPhotonsDz02Reg
                trigger_bits=64,
            ),
            tau2=TriggerLeg(
                pdg_id=15,
                # min_pt=40.0,
                # filter names:
                # hltDoublePFTau35TrackPt1TightChargedIsolationAndTightOOSCPhotonsDz02Reg
                trigger_bits=64,
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data),
        tags={"cross_trigger", "cross_tau_tau"},
    )
    config.x.triggers.add(
        name="HLT_DoubleMediumChargedIsoPFTau40_Trk1_TightID_eta2p1_Reg",
        id=505,
        legs=dict(
            tau1=TriggerLeg(
                pdg_id=15,
                # min_pt=45.0,
                # filter names:
                # hltDoublePFTau40TrackPt1MediumChargedIsolationAndTightOOSCPhotonsDz02Reg
                trigger_bits=64,
            ),
            tau2=TriggerLeg(
                pdg_id=15,
                # min_pt=45.0,
                # filter names:
                # hltDoublePFTau40TrackPt1MediumChargedIsolationAndTightOOSCPhotonsDz02Reg
                trigger_bits=64,
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data),
        tags={"cross_trigger", "cross_tau_tau"},
    )
    config.x.triggers.add(
        name="HLT_DoubleTightChargedIsoPFTau40_Trk1_eta2p1_Reg",
        id=506,
        legs=dict(
            tau1=TriggerLeg(
                pdg_id=15,
                # min_pt=45.0,
                # filter names:
                # hltDoublePFTau40TrackPt1TightChargedIsolationDz02Reg
                trigger_bits=64,
            ),
            tau2=TriggerLeg(
                pdg_id=15,
                # min_pt=45.0,
                # filter names:
                # hltDoublePFTau40TrackPt1TightChargedIsolationDz02Reg
                trigger_bits=64,
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data),
        tags={"cross_trigger", "cross_tau_tau"},
    )

    #
    # vbf
    #
    config.x.triggers.add(
        name="HLT_VBF_DoubleLooseChargedIsoPFTau20_Trk1_eta2p1_Reg",
        id=601,
        legs=dict(
            tau1=TriggerLeg(
                pdg_id=15,
                # min_pt=25.0,
                # filter names:
                # hltDoublePFTau20TrackPt1LooseChargedIsolation
                trigger_bits=2048,
            ),
            tau2=TriggerLeg(
                pdg_id=15,
                # min_pt=25.0,
                # filter names:
                # hltDoublePFTau20TrackPt1LooseChargedIsolation
                trigger_bits=2048,
            ),
            # additional leg infos for vbf jets
            vbf1=TriggerLeg(
                # min_pt=115.0,
                # filter names:
                # hltMatchedVBFOnePFJet2CrossCleanedFromDoubleLooseChargedIsoPFTau20
                trigger_bits=1,
            ),
            vbf2=TriggerLeg(
                # min_pt=40.0,
                # filter names:
                # hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleLooseChargedIsoPFTau20
                trigger_bits=1,
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "D"),
        tags={"cross_trigger", "cross_tau_tau_vbf"},
    )


def add_triggers_2018(config: od.Config) -> None:
    config.x.triggers = od.UniqueObjectIndex(Trigger)

    #
    # single electron
    #
    config.x.triggers.add(
        name="HLT_Ele32_WPTight_Gsf",
        id=202,
        legs=dict(
            e=TriggerLeg(
                pdg_id=11,
                # min_pt=35.0,
                # filter names:
                # hltEle32WPTightGsfTrackIsoFilter
                trigger_bits=2,
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "D"),
        tags={"single_trigger", "single_e"},
    )
    config.x.triggers.add(
        name="HLT_Ele35_WPTight_Gsf",
        id=204,
        legs=dict(
            e=TriggerLeg(
                pdg_id=11,
                # min_pt=38.0,
                # filter names:
                # hltEle35noerWPTightGsfTrackIsoFilter
                trigger_bits=2,
            ),
        ),
        tags={"single_trigger", "single_e"},
    )

    #
    # single muon
    #
    config.x.triggers.add(
        name="HLT_IsoMu24",
        id=105,
        legs=dict(
            mu=TriggerLeg(
                pdg_id=13,
                # min_pt=26.0,
                # filter names:
                # hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p07
                trigger_bits=2,
            ),
        ),
        tags={"single_trigger", "single_mu"},
    )
    config.x.triggers.add(
        name="HLT_IsoMu27",
        id=106,
        legs=dict(
            mu=TriggerLeg(
                pdg_id=13,
                # min_pt=29.0,
                # filter names:
                # hltL3crIsoL1sMu22Or25L1f0L2f10QL3f27QL3trkIsoFiltered0p07
                trigger_bits=2,
            ),
        ),
        tags={"single_trigger", "single_mu"},
    )

    #
    # e tauh
    #
    config.x.triggers.add(
        name="HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTau30_eta2p1_CrossL1",
        id=404,
        legs=dict(
            e=TriggerLeg(
                pdg_id=11,
                # min_pt=27.0,
                # filter names:
                # hltEle24erWPTightGsfTrackIsoFilterForTau
                # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
                trigger_bits=2 + 64,
            ),
            tau=TriggerLeg(
                pdg_id=15,
                # min_pt=35.0,
                # filter names:
                # hltSelectedPFTau30LooseChargedIsolationL1HLTMatched
                # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
                trigger_bits=1024 + 256,
            ),
        ),
        # the non-HPS path existed only for data and is fully covered in MC below
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data),
        tags={"cross_trigger", "cross_e_tau"},
    )

    #
    # mu tauh
    #
    config.x.triggers.add(
        name="HLT_IsoMu20_eta2p1_LooseChargedIsoPFTau27_eta2p1_CrossL1",
        id=303,
        legs=dict(
            mu=TriggerLeg(
                pdg_id=13,
                # min_pt=22.0,
                # filter names:
                # hltL3crIsoL1sMu18erTau24erIorMu20erTau24erL1f0L2f10QL3f20QL3trkIsoFiltered0p07
                # hltOverlapFilterIsoMu20LooseChargedIsoPFTau27L1Seeded
                trigger_bits=2 + 64,
            ),
            tau=TriggerLeg(
                pdg_id=15,
                # min_pt=32.0,
                # filter names:
                # hltSelectedPFTau27LooseChargedIsolationAgainstMuonL1HLTMatched or
                # hltOverlapFilterIsoMu20LooseChargedIsoPFTau27L1Seeded
                trigger_bits=1024 + 512,
            ),
        ),
        # the non-HPS path existed only for data and is fully covered in MC below
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data),
        tags={"cross_trigger", "cross_mu_tau"},
    )

    #
    # tauh tauh
    #
    config.x.triggers.add(
        name="HLT_DoubleTightChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg",
        id=504,
        legs=dict(
            tau1=TriggerLeg(
                pdg_id=15,
                # min_pt=40.0,
                # filter names:
                # hltDoublePFTau35TrackPt1TightChargedIsolationAndTightOOSCPhotonsDz02Reg
                trigger_bits=64,
            ),
            tau2=TriggerLeg(
                pdg_id=15,
                # min_pt=40.0,
                # filter names:
                # hltDoublePFTau35TrackPt1TightChargedIsolationAndTightOOSCPhotonsDz02Reg
                trigger_bits=64,
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data),
        tags={"cross_trigger", "cross_tau_tau"},
    )
    config.x.triggers.add(
        name="HLT_DoubleMediumChargedIsoPFTau40_Trk1_TightID_eta2p1_Reg",
        id=505,
        legs=dict(
            tau1=TriggerLeg(
                pdg_id=15,
                # min_pt=45.0,
                # filter names:
                # hltDoublePFTau40TrackPt1MediumChargedIsolationAndTightOOSCPhotonsDz02Reg
                trigger_bits=64,
            ),
            tau2=TriggerLeg(
                pdg_id=15,
                # min_pt=45.0,
                # filter names:
                # hltDoublePFTau40TrackPt1MediumChargedIsolationAndTightOOSCPhotonsDz02Reg
                trigger_bits=64,
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data),
        tags={"cross_trigger", "cross_tau_tau"},
    )
    config.x.triggers.add(
        name="HLT_DoubleTightChargedIsoPFTau40_Trk1_eta2p1_Reg",
        id=506,
        legs=dict(
            tau1=TriggerLeg(
                pdg_id=15,
                # min_pt=45.0,
                # filter names:
                # hltDoublePFTau40TrackPt1TightChargedIsolationDz02Reg
                trigger_bits=64,
            ),
            tau2=TriggerLeg(
                pdg_id=15,
                # min_pt=45.0,
                # filter names:
                # hltDoublePFTau40TrackPt1TightChargedIsolationDz02Reg
                trigger_bits=64,
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data),
        tags={"cross_trigger", "cross_tau_tau"},
    )

    #
    # vbf
    #
    config.x.triggers.add(
        name="HLT_VBF_DoubleLooseChargedIsoPFTau20_Trk1_eta2p1_Reg",
        id=601,
        legs=dict(
            tau1=TriggerLeg(
                pdg_id=15,
                # min_pt=25.0,
                # filter names:
                # hltDoublePFTau20TrackPt1LooseChargedIsolation
                trigger_bits=2048,
            ),
            tau2=TriggerLeg(
                pdg_id=15,
                # min_pt=25.0,
                # filter names:
                # hltDoublePFTau20TrackPt1LooseChargedIsolation
                trigger_bits=2048,
            ),
            # additional leg infos for vbf jets
            vbf1=TriggerLeg(
                # min_pt=115.0,
                # filter names:
                # hltMatchedVBFOnePFJet2CrossCleanedFromDoubleLooseChargedIsoPFTau20
                trigger_bits=1,
            ),
            vbf2=TriggerLeg(
                # min_pt=40.0,
                # filter names:
                # hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleLooseChargedIsoPFTau20
                trigger_bits=1,
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "D"),
        tags={"cross_trigger", "cross_tau_tau_vbf"},
    )


def add_triggers_2022(config: od.Config) -> None:
    """
    Adds all triggers to a *config*. For the conversion from filter names to trigger bits, see
    https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py.
    Tau Trigger: https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauTrigger#Trigger_Table_for_2022
    Electron Trigger: https://twiki.cern.ch/twiki/bin/view/CMS/EgHLTRunIIISummary
    Muon Trigger: https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLT2022
    """
    # get the nano version
    nano_trigger_bit_version = config.x("nano_trigger_bit_version", config.campaign.x.version)
    get_bit_sum_v = functools.partial(get_bit_sum, nano_trigger_bit_version)

    config.x.triggers = od.UniqueObjectIndex(Trigger)

    #
    # single electron
    #
    config.x.triggers.add(
        name="HLT_Ele30_WPTight_Gsf",
        id=205,
        legs=dict(
            e=TriggerLeg(
                pdg_id=11,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltEle30WPTightGsfTrackIsoFilter
                trigger_bits=get_bit_sum_v("e", [
                    "WPTightTrackIso",
                ]),
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: (
            dataset_inst.is_mc or
            dataset_inst.has_tag("etau") or
            dataset_inst.has_tag("ee") or
            dataset_inst.has_tag("emu_from_e") or
            dataset_inst.has_tag("emu_from_mu")
        )),
        tags={"single_trigger", "single_e"},
    )

    #
    # single muon
    #
    config.x.triggers.add(
        name="HLT_IsoMu24",
        id=105,
        legs=dict(
            mu=TriggerLeg(
                pdg_id=13,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p08
                trigger_bits=get_bit_sum_v("mu", [
                    "Iso",
                    "SingleMuon",
                ]),
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: (
            dataset_inst.is_mc or
            dataset_inst.has_tag("mutau") or
            dataset_inst.has_tag("emu_from_e") or
            dataset_inst.has_tag("emu_from_mu") or
            dataset_inst.has_tag("mumu")
        )),
        tags={"single_trigger", "single_mu"},
    )

    #
    # e tauh
    #
    config.x.triggers.add(
        name="HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
        id=405,
        legs=dict(
            e=TriggerLeg(
                pdg_id=11,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltHpsOverlapFilterIsoEle24WPTightGsfLooseETauWPDeepTauPFTau30
                trigger_bits=get_bit_sum_v("e", [
                    "OverlapFilterPFTau",
                    "EleTau",
                ]),
            ),
            tau=TriggerLeg(
                pdg_id=15,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltHpsOverlapFilterIsoEle24WPTightGsfLooseETauWPDeepTauPFTau30
                trigger_bits=get_bit_sum_v("tau", [
                    "DeepTau",
                    "HPS",
                    "OverlapFilterIsoEle",
                    "ETauFilter" if nano_trigger_bit_version in {14, 15} else None,
                ]),
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.has_tag("etau")),
        tags={"cross_trigger", "cross_e_tau"},
    )

    #
    # mu tauh
    #
    config.x.triggers.add(
        name="HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1",
        id=304,
        legs=dict(
            mu=TriggerLeg(
                pdg_id=13,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltHpsOverlapFilterIsoMu20LooseMuTauWPDeepTauPFTau27L1Seeded
                trigger_bits=get_bit_sum_v("mu", [
                    "OverlapFilterPFTau",
                    "MuTau",
                ]),
            ),
            tau=TriggerLeg(
                pdg_id=15,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltHpsOverlapFilterIsoMu20LooseMuTauWPDeepTauPFTau27L1Seeded
                trigger_bits=get_bit_sum_v("tau", [
                    "DeepTau",
                    "HPS",
                    "OverlapFilterIsoMu",
                    "MuTauFilter" if nano_trigger_bit_version in {14, 15} else None,
                    "MatchL1HLT",
                ]),
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.has_tag("mutau")),
        tags={"cross_trigger", "cross_mu_tau"},
    )

    #
    # tauh tauh
    #
    config.x.triggers.add(
        name="HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1",
        id=507,
        legs=dict(
            tau1=TriggerLeg(
                pdg_id=15,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltHpsDoublePFTau35MediumDitauWPDeepTauL1HLTMatched
                trigger_bits=get_bit_sum_v("tau", [
                    "DeepTauDiTau",
                    "HPS",
                    "Medium" if nano_trigger_bit_version in {14, 15} else None,
                ]),
            ),
            tau2=TriggerLeg(
                pdg_id=15,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltHpsDoublePFTau35MediumDitauWPDeepTauL1HLTMatched
                trigger_bits=get_bit_sum_v("tau", [
                    "DeepTauDiTau",
                    "HPS",
                    "Medium" if nano_trigger_bit_version in {14, 15} else None,
                ]),
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.has_tag("tautau")),
        tags={"cross_trigger", "cross_tau_tau"},
    )

    #
    # vbf
    #
    if nano_trigger_bit_version >= 14:
        config.x.triggers.add(
            name="HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1",
            id=602,
            legs=dict(
                tau1=TriggerLeg(
                    pdg_id=15,
                    # min_pt=None,  # cut on reco objects, not TrigObj
                    # filter names:
                    # hltHpsDoublePFTau20TrackDeepTauDitauWPForVBFIsoTau
                    # HPS and DeepTau actually redundant for v14 but needed for v12
                    # as there is nothing else matching due to wrong VBFpDoublePFTau_run3 bit
                    trigger_bits=get_bit_sum_v("tau", [
                        "VBFDiTau" if nano_trigger_bit_version in {14, 15} else None,
                        "HPS",
                        "DeepTau",
                    ]),
                ),
                tau2=TriggerLeg(
                    pdg_id=15,
                    # min_pt=None,  # cut on reco objects, not TrigObj
                    # filter names:
                    # hltHpsDoublePFTau20TrackDeepTauDitauWPForVBFIsoTau
                    trigger_bits=get_bit_sum_v("tau", [
                        "VBFDiTau" if nano_trigger_bit_version in {14, 15} else None,
                        "HPS",
                        "DeepTau",
                    ]),
                ),
                # additional leg infos for vbf jets
                vbf1=TriggerLeg(
                    pdg_id=1,
                    # min_pt=None,  # cut on reco objects, not TrigObj
                    # filter names:
                    # hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20?
                    trigger_bits=get_bit_sum_v("jet", [
                        "VBFcrossCleanedDeepTauPFTau" if nano_trigger_bit_version in {14, 15} else None,
                    ]),
                ),
                vbf2=TriggerLeg(
                    pdg_id=1,
                    # min_pt=None,  # cut on reco objects, not TrigObj
                    # filter names:
                    # hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20?
                    trigger_bits=get_bit_sum_v("jet", [
                        "VBFcrossCleanedDeepTauPFTau" if nano_trigger_bit_version in {14, 15} else None,
                    ]),
                ),
            ),
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.has_tag("tautau")),
            tags={"cross_trigger", "cross_tau_tau_vbf"},
        )

    # Currently disabled since it may not be needed
    # config.x.triggers.add(
    #     name="HLT_DoublePFJets40_Mass500_MediumDeepTauPFTauHPS45_L2NN_MediumDeepTauPFTauHPS20_eta2p1",
    #     id=604,
    #     legs=dict(
    #         TriggerLeg(
    #             pdg_id=15,
    #             # min_pt=25.0,
    #             trigger_bits=None,
    #         ),
    #         TriggerLeg(
    #             pdg_id=15,
    #             # min_pt=25.0,
    #             # filter names:
    #             trigger_bits=None,
    #         )
    #     ],
    # )

    #
    # tau tau jet
    #
    config.x.triggers.add(
        name="HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60",
        id=701,
        legs=dict(
            tau1=TriggerLeg(
                pdg_id=15,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltHpsOverlapFilterDeepTauDoublePFTau30PFJet60
                trigger_bits=get_bit_sum_v("tau", [
                    "DiTauAndPFJet",
                ]),
            ),
            tau2=TriggerLeg(
                pdg_id=15,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltHpsOverlapFilterDeepTauDoublePFTau30PFJet60
                trigger_bits=get_bit_sum_v("tau", [
                    "DiTauAndPFJet",
                ]),
            ),
            jet=TriggerLeg(
                pdg_id=1,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltHpsOverlapFilterDeepTauDoublePFTau30PFJet60
                trigger_bits=get_bit_sum_v("jet", [
                    "DoubleTau+Jet",
                ]),
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.has_tag("tautau")),
        tags={"cross_trigger", "cross_tau_tau_jet"},
    )


def add_triggers_2023(config: od.Config) -> None:
    """
    Adds all triggers to a *config*. For the conversion from filter names to trigger bits, see
    https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py.
    """
    # get trigger bits for the requested nano version
    nano_trigger_bit_version = config.x("nano_trigger_bit_version", config.campaign.x.version)
    get_bit_sum_v = functools.partial(get_bit_sum, nano_trigger_bit_version)

    campaign_postfix = "preBPix" if config.campaign.has_tag("preBPix") else "postBPix"

    config.x.triggers = od.UniqueObjectIndex(Trigger)

    #
    # single electron
    #
    config.x.triggers.add(
        name="HLT_Ele30_WPTight_Gsf",
        id=205,
        legs=dict(
            e=TriggerLeg(
                pdg_id=11,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # WPTightTrackIso
                trigger_bits=get_bit_sum_v("e", [
                    "WPTightTrackIso",
                ]),
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: (
            dataset_inst.is_mc or
            dataset_inst.has_tag("etau") or
            dataset_inst.has_tag("ee") or
            dataset_inst.has_tag("emu_from_e") or
            dataset_inst.has_tag("emu_from_mu") or
            dataset_inst.has_tag("parking_vbf")
        )),
        tags={"single_trigger", "single_e"},
    )

    #
    # single muon
    #
    config.x.triggers.add(
        name="HLT_IsoMu24",
        id=105,
        legs=dict(
            mu=TriggerLeg(
                pdg_id=13,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p08 (1mu + Iso)
                trigger_bits=get_bit_sum_v("mu", [
                    "Iso",
                    "SingleMuon",
                ]),
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: (
            dataset_inst.is_mc or
            dataset_inst.has_tag("mutau") or
            dataset_inst.has_tag("emu_from_e") or
            dataset_inst.has_tag("emu_from_mu") or
            dataset_inst.has_tag("mumu") or
            dataset_inst.has_tag("parking_vbf")
        )),
        tags={"single_trigger", "single_mu"},
    )

    #
    # e tauh
    #
    config.x.triggers.add(
        name="HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
        id=405,
        legs=dict(
            e=TriggerLeg(
                pdg_id=11,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltHpsOverlapFilterIsoEle24WPTightGsfLooseETauWPDeepTauPFTau30 (OverlapFilter)
                trigger_bits=get_bit_sum_v("e", [
                    "OverlapFilterPFTau",
                    "EleTau",
                ]),
            ),
            tau=TriggerLeg(
                pdg_id=15,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltHpsOverlapFilterIsoEle24WPTightGsfLooseETauWPDeepTauPFTau30
                trigger_bits=get_bit_sum_v("tau", [
                    "DeepTau",
                    "HPS",
                    "OverlapFilterIsoEle",
                    "ETauFilter" if nano_trigger_bit_version in {14, 15} else None,
                ]),
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: (
            dataset_inst.is_mc or
            dataset_inst.has_tag("etau") or
            dataset_inst.has_tag("parking_vbf")
        )),
        tags={"cross_trigger", "cross_e_tau"},
    )

    #
    # mu tauh
    #
    config.x.triggers.add(
        name="HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1",
        id=304,
        legs=dict(
            mu=TriggerLeg(
                pdg_id=13,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltHpsOverlapFilterIsoMu20LooseMuTauWPDeepTauPFTau27L1Seeded (OverlapFilter PFTau)
                trigger_bits=get_bit_sum_v("mu", [
                    "OverlapFilterPFTau",
                    "MuTau",
                ]),
            ),
            tau=TriggerLeg(
                pdg_id=15,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltHpsSelectedPFTau27LooseMuTauWPDeepTauVsJetsAgainstMuonL1HLTMatched (DeepTau + HPS)
                trigger_bits=get_bit_sum_v("tau", [
                    "DeepTau",
                    "HPS",
                    "OverlapFilterIsoMu",
                    "MuTauFilter" if nano_trigger_bit_version in {14, 15} else None,
                    "MatchL1HLT",
                ]),
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: (
            dataset_inst.is_mc or
            dataset_inst.has_tag("mutau") or
            dataset_inst.has_tag("parking_vbf")
        )),
        tags={"cross_trigger", "cross_mu_tau"},
    )

    #
    # tauh tauh
    #
    config.x.triggers.add(
        name="HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1",
        id=507,
        legs=dict(
            tau1=TriggerLeg(
                pdg_id=15,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltHpsDoublePFTau35MediumDitauWPDeepTauL1HLTMatched (Deeptau + HPS)
                trigger_bits=get_bit_sum_v("tau", [
                    "DeepTauDiTau",
                    "HPS",
                    "Medium" if nano_trigger_bit_version in {14, 15} else None,
                ]),
            ),
            tau2=TriggerLeg(
                pdg_id=15,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltHpsDoublePFTau35MediumDitauWPDeepTauL1HLTMatched (Deeptau + HPS)
                trigger_bits=get_bit_sum_v("tau", [
                    "DeepTauDiTau",
                    "HPS",
                    "Medium" if nano_trigger_bit_version in {14, 15} else None,
                ]),
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: (
            dataset_inst.is_mc or
            dataset_inst.has_tag("tautau") or
            dataset_inst.has_tag("parking_vbf")
        )),
        tags={"cross_trigger", "cross_tau_tau"},
    )

    #
    # vbf
    #
    config.x.triggers.add(
        name="HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1",
        id=602,
        legs=dict(
            tau1=TriggerLeg(
                pdg_id=15,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltHpsDoublePFTau20TrackDeepTauDitauWPForVBFIsoTau
                trigger_bits=get_bit_sum_v("tau", [
                    "VBFDiTau" if nano_trigger_bit_version in {14, 15} else None,
                    "HPS",
                    "DeepTau",
                ]),
            ),
            tau2=TriggerLeg(
                pdg_id=15,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltHpsDoublePFTau20TrackDeepTauDitauWPForVBFIsoTau
                trigger_bits=get_bit_sum_v("tau", [
                    "VBFDiTau" if nano_trigger_bit_version in {14, 15} else None,
                    "HPS",
                    "DeepTau",
                ]),
            ),
            # additional leg infos for vbf jets
            vbf1=TriggerLeg(
                pdg_id=1,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20?
                trigger_bits=get_bit_sum_v("jet", [
                    "VBFcrossCleanedDeepTauPFTau" if nano_trigger_bit_version in {14, 15} else None,
                ]),
            ),
            vbf2=TriggerLeg(
                pdg_id=1,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20?
                trigger_bits=get_bit_sum_v("jet", [
                    "VBFcrossCleanedDeepTauPFTau" if nano_trigger_bit_version in {14, 15} else None,
                ]),
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: ((
            (dataset_inst.is_mc or dataset_inst.has_tag("tautau")) and
            campaign_postfix == "preBPix"
        ) or (
            (dataset_inst.is_mc or dataset_inst.has_tag("parking_vbf")) and
            campaign_postfix == "postBPix"
        ))),
        tags={"cross_trigger", "cross_tau_tau_vbf"},
    )

    config.x.triggers.add(
        name="HLT_VBF_DiPFJet105_40_Mjj1000_Detajj3p5",
        id=605,
        legs=dict(
            vbf1=TriggerLeg(
                pdg_id=1,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltL1PFJetCategoriesVBFinclLoose
                trigger_bits=get_bit_sum_v("jet", [
                    "VBFincl" if nano_trigger_bit_version == 15 else None,
                ]),
            ),
            vbf2=TriggerLeg(
                pdg_id=1,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltL1PFJetCategoriesVBFinclLoose
                trigger_bits=get_bit_sum_v("jet", [
                    "VBFincl" if nano_trigger_bit_version == 15 else None,
                ]),
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.has_tag("parking_vbf")),
        tags={"cross_trigger", "cross_vbf"},
    )

    config.x.triggers.add(
        name="HLT_VBF_DiPFJet90_40_Mjj600_Detajj2p5_Mu3_TrkIsoVVL",
        id=606,
        legs=dict(
            vbf1=TriggerLeg(
                pdg_id=1,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltL1PFJetCategoriesVBFMu
                trigger_bits=get_bit_sum_v("jet", [
                    "VBFmu" if nano_trigger_bit_version == 15 else None,
                ]),
            ),
            vbf2=TriggerLeg(
                pdg_id=1,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltL1PFJetCategoriesVBFMu
                trigger_bits=get_bit_sum_v("jet", [
                    "VBFmu" if nano_trigger_bit_version == 15 else None,
                ]),
            ),
            mu=TriggerLeg(
                pdg_id=13,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltMuon3RelTrkIsoVVLFiltered -> bit 0, "*RelTrkIsoVVLFiltered", "TrkIsoVVL"
                trigger_bits=get_bit_sum_v("mu", [
                    "TrkIsoVVL",  # -> CCLUB has none here on 05.08.2025
                ]),
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.has_tag("parking_vbf")),
        tags={"cross_trigger", "cross_mu_vbf"},
    )

    config.x.triggers.add(
        name="HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Ele12_eta2p1_WPTight_Gsf",
        id=607,
        legs=dict(
            vbf1=TriggerLeg(
                pdg_id=1,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltOverlapFilterDoublePFJet45Ele12
                trigger_bits=get_bit_sum_v("jet", [
                    "VBFele" if nano_trigger_bit_version == 15 else None,
                ]),
            ),
            vbf2=TriggerLeg(
                pdg_id=1,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltOverlapFilterDoublePFJet45Ele12
                trigger_bits=get_bit_sum_v("jet", [
                    "VBFele" if nano_trigger_bit_version == 15 else None,
                ]),
            ),
            e=TriggerLeg(
                pdg_id=11,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltEle12erWPTightGsfTrackIsoFilterNoRhoCorrectionForVBF
                trigger_bits=get_bit_sum_v("e", [
                    "VBFWPTightGsfTrackIso" if nano_trigger_bit_version == 15 else None,
                ]),
            ),
        ),
        applies_to_dataset=(
            (lambda dataset_inst: (dataset_inst.is_mc or dataset_inst.has_tag("parking_vbf")) and
            campaign_postfix == "preBPix")
        ),
        tags={"cross_trigger", "cross_e_vbf"},
    )

    config.x.triggers.add(
        name="HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Ele17_eta2p1_WPTight_Gsf",
        id=608,
        legs=dict(
            vbf1=TriggerLeg(
                pdg_id=1,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltOverlapFilterDoublePFJet45Ele17
                trigger_bits=get_bit_sum_v("jet", [
                    "VBFele" if nano_trigger_bit_version == 15 else None,
                ]),
            ),
            vbf2=TriggerLeg(
                pdg_id=1,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltOverlapFilterDoublePFJet45Ele17
                trigger_bits=get_bit_sum_v("jet", [
                    "VBFele" if nano_trigger_bit_version == 15 else None,
                ]),
            ),
            e=TriggerLeg(
                pdg_id=11,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltEle17erWPTightGsfTrackIsoFilterNoRhoCorrectionForVBF
                trigger_bits=get_bit_sum_v("e", [
                    "VBFWPTightGsfTrackIso" if nano_trigger_bit_version == 15 else None,
                ]),
            ),
        ),
        applies_to_dataset=(
            (lambda dataset_inst: (dataset_inst.is_mc or dataset_inst.has_tag("parking_vbf")) and
            campaign_postfix == "postBPix")
        ),
        tags={"cross_trigger", "cross_e_vbf"},
    )

    config.x.triggers.add(
        name="HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_MediumDeepTauPFTauHPS45_L2NN_eta2p1",
        id=609,
        legs=dict(
            vbf1=TriggerLeg(
                pdg_id=1,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltMatchedVBFIsoTauTwoPFJets2CrossCleanedUsingDiJetCorrCheckerWithMediumDiTauSingleTauHLT?
                # CCLUB has VBFcrossCleanedDijet -> 19
                trigger_bits=get_bit_sum_v("jet", [
                    "VBFcrossCleanedUsingDijetCorr",

                ]),
            ),
            vbf2=TriggerLeg(
                pdg_id=1,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltMatchedVBFIsoTauTwoPFJets2CrossCleanedUsingDiJetCorrCheckerWithMediumDiTauSingleTauHLT?
                trigger_bits=get_bit_sum_v("jet", [
                    "VBFcrossCleanedUsingDijetCorr",
                ]),
            ),
            tau=TriggerLeg(
                pdg_id=15,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltHpsSinglePFTau45MediumDitauWPDeepTauL1HLTMatchedSingleTauHLT?
                # CCLUB has DeepTau, VBFSingleTau -> 3, 30
                # redundant with VBFSingleTau
                trigger_bits=get_bit_sum_v("tau", [
                    "VBFSingleTau" if nano_trigger_bit_version in {14, 15} else None,
                    # TODO: add trigger bits for v12
                ]),
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.has_tag("parking_vbf")),
        tags={"cross_trigger", "cross_tau_vbf"},
    )

    #
    # tau tau jet
    #
    config.x.triggers.add(
        name="HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60",
        id=701,
        legs=dict(
            tau1=TriggerLeg(
                pdg_id=15,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltHpsDoublePFTau30MediumDitauWPDeepTauL1HLTMatchedDoubleTauJet
                trigger_bits=get_bit_sum_v("tau", [
                    "DiTauAndPFJet",
                ]),
            ),
            tau2=TriggerLeg(
                pdg_id=15,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltHpsDoublePFTau30MediumDitauWPDeepTauL1HLTMatchedDoubleTauJet
                trigger_bits=get_bit_sum_v("tau", [
                    "DiTauAndPFJet",
                ]),
            ),
            jet=TriggerLeg(
                pdg_id=1,
                # min_pt=None,  # cut on reco objects, not TrigObj
                # filter names:
                # hltHpsOverlapFilterDeepTauDoublePFTau30PFJet60
                trigger_bits=get_bit_sum_v("jet", [
                    "DoubleTau+Jet",
                ]),
            ),
        ),
        applies_to_dataset=(lambda dataset_inst: (
            dataset_inst.is_mc or
            dataset_inst.has_tag("tautau") or
            dataset_inst.has_tag("parking_vbf")
        )),
        tags={"cross_trigger", "cross_tau_tau_jet"},
    )
