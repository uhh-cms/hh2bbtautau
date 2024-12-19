# coding: utf-8

"""
Definition of triggers

General requirement from the lepton selection:
For cross triggers, the lepton leg (lepton= {"e", "mu"}) must be defined before the tau leg.
An error here would be caught in the lepton selection, but it is better to avoid it.
# TODO: add name to the TriggerLeg class to make it independent from the ordering of the legs

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
"HLT_IsoMu22"
id=101
"HLT_IsoMu22_eta2p1"
id=102
"HLT_IsoTkMu22"
id=103
"HLT_IsoTkMu22_eta2p1"
id=104
"HLT_IsoMu24"
id=105
"HLT_IsoMu27"
id=106

"HLT_Ele25_eta2p1_WPTight_Gsf"
id=201
"HLT_Ele32_WPTight_Gsf"
id=202
"HLT_Ele32_WPTight_Gsf_L1DoubleEG"
id=203
"HLT_Ele35_WPTight_Gsf"
id=204
"HLT_Ele30_WPTight_Gsf"
id=205

"HLT_IsoMu19_eta2p1_LooseIsoPFTau20"
id=301
"HLT_IsoMu19_eta2p1_LooseIsoPFTau20_SingleL1"
id=302
"HLT_IsoMu20_eta2p1_LooseChargedIsoPFTau27_eta2p1_CrossL1"
id=303
"HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1"
id=304

"HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_SingleL1"
id=401
"HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau20"
id=402
"HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau30"
id=403
"HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTau30_eta2p1_CrossL1"
id=404
"HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1"
id=405

"HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Reg"
id=501
"HLT_DoubleMediumCombinedIsoPFTau35_Trk1_eta2p1_Reg"
id=502
"HLT_DoubleMediumChargedIsoPFTau35_Trk1_eta2p1_Reg"
id=503
"HLT_DoubleTightChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg"
id=504
"HLT_DoubleMediumChargedIsoPFTau40_Trk1_TightID_eta2p1_Reg"
id=505
"HLT_DoubleTightChargedIsoPFTau40_Trk1_eta2p1_Reg"
id=506
"HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1"
id=507
"HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1"
id=508
"HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1"
id=509

"HLT_VBF_DoubleLooseChargedIsoPFTau20_Trk1_eta2p1_Reg"
id=601
"HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1"
id=602
"HLT_VBF_DoubleLooseChargedIsoPFTauHPS20_Trk1_eta2p1"
id=603
"HLT_DoublePFJets40_Mass500_MediumDeepTauPFTauHPS45_L2NN_MediumDeepTauPFTauHPS20_eta2p1"
id=604

"HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60"
id=701
"HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet75"
id=702
"""

import order as od

from hbt.config.util import Trigger, TriggerLeg

# TODO: better use the name or the Trig_obj_id for the keys?
# Use the CCLub names for the trigger bits and improve them when necessary
trigger_bit_matching = {
    # checked with https://github.com/cms-sw/cmssw/blob/CMSSW_13_0_X/PhysicsTools/NanoAOD/python/triggerObjects_cff.py
    # and in particular https://github.com/cms-sw/cmssw/blob/2defd844e96613d2438b690d10b79c773e02ab57/PhysicsTools/NanoAOD/python/triggerObjects_cff.py  # noqa
    "NanoAODv12": {
        # electron legs
        "Electron": {
            "CaloIdLTrackIdLIsoVL": 1,  # 0
            "WPTightTrackIso": 2,  # 1
            "WPLooseTrackIso": 4,  # 2
            "OverlapFilterPFTau": 8,  # 3
            "DiElectron": 16,  # 4
            "MuEle": 32,  # 5
            "EleTau": 64,  # 6
            "TripleElectron": 128,  # 7
            "SingleMuonDiEle": 256,  # 8
            "DiMuonSingleEle": 512,  # 9
            "SingleEle_L1DoubleAndSingleEle": 1024,  # 10
            "SingleEle_CaloIdVT_GsfTrkIdT": 2048,  # 11
            "SingleEle_PFJet": 4096,  # 12
            "Photon175_Photon200": 8192,  # 13
        },
        # muon legs
        "Muon": {
            "TrkIsoVVL": 1,  # 0
            "Iso": 2,  # 1
            "OverlapFilterPFTau": 4,  # 2
            "SingleMuon": 8,  # 3
            "DiMuon": 16,  # 4
            "MuEle": 32,  # 5
            "MuTau": 64,  # 6
            "TripleMuon": 128,  # 7
            "DiMuonSingleEle": 256,  # 8
            "SingleMuonDiEle": 512,  # 9
            "Mu50": 1024,  # 10
            "Mu100": 2048,  # 11
            "SingleMuonSinglePhoton": 4096,  # 12
        },
        # tau legs
        "Tau": {
            "LooseChargedIso": 1,  # 0
            "MediumChargedIso": 2,  # 1
            "TightChargedIso": 4,  # 2
            "DeepTau": 8,  # 3
            "TightOOSCPhotons": 16,  # 4
            "Hps": 32,  # 5
            "ChargedIsoDiTau": 64,  # 6
            "DeeptauDiTau": 128,  # 7
            "OverlapFilterIsoEle": 256,  # 8
            "OverlapFilterIsoMu": 512,  # 9
            "SingleTauOrTauMet": 1024,  # 10
            "VBFpDoublePFTau_run2": 2048,  # 11
            "VBFpDoublePFTau_run3": 4096,  # 12
            "DiPFJetAndDiTau": 8192,  # 13
            "DiTauAndPFJet": 16384,  # 14
            "DisplacedTau": 32768,  # 15
            "Monitoring": 65536,  # 16
            "RegionalPaths": 131072,  # 17
            "L1SeededPaths": 262144,  # 18
            "1Prong": 524288,  # 19
        },
        # jet legs
        "Jet": {
            "4PixelOnlyPFCentralJetTightIDPt20": 1,  # 0
            "3PixelOnlyPFCentralJetTightIDPt30": 2,  # 1
            "PFJetFilterTwoC30": 4,  # 2
            "4PFCentralJetTightIDPt30": 8,  # 3
            "4PFCentralJetTightIDPt35": 16,  # 4
            "QuadCentralJet30": 32,  # 5
            "2PixelOnlyPFCentralJetTightIDPt40": 64,  # 6
            "L1sTripleJetVBF_orHTT_orDoubleJet_orSingleJet": 128,  # 7
            "3PFCentralJetTightIDPt40": 256,  # 8
            "3PFCentralJetTightIDPt45": 512,  # 9
            "L1sQuadJetsHT": 1024,  # 10
            "BTagCaloDeepCSVp17Double": 2048,  # 11
            "PFCentralJetLooseIDQuad30": 4096,  # 12
            "1PFCentralJetLooseID75": 8192,  # 13
            "2PFCentralJetLooseID60": 16384,  # 14
            "3PFCentralJetLooseID45": 32768,  # 15
            "4PFCentralJetLooseID40": 65536,  # 16
            "DoubleTau+Jet": 131072,  # 17
            "VBFcrossCleanedDeepTauPFTau": 262144,  # 18  TODO: change name? idea comes from v14
            "VBFcrossCleanedUsingDijetCorr": 524288,  # 19  TODO: change name? idea comes from v14
            "MonitoringMuon+Tau+Jet": 1048576,  # 20
            "2PFCentralJetTightIDPt50": 2097152,  # 21
            "1PixelOnlyPFCentralJetTightIDPt60": 4194304,  # 22
            "1PFCentralJetTightIDPt70": 8388608,  # 23
            "BTagPFDeepJet1p5Single": 16777216,  # 24
            "BTagPFDeepJet4p5Triple": 33554432,  # 25
            "2BTagSumOR2BTagMeanPaths": 67108864,  # 26
            "2/1PixelOnlyPFCentralJetTightIDPt20/50": 134217728,  # 27
            "2PFCentralJetTightIDPt30": 268435456,  # 28
            "1PFCentralJetTightIDPt60": 536870912,  # 29
            "PF2CentralJetPt30PNet2BTagMean0p50": 1073741824,  # 30
        },
    },
    # v13 not done, pb e.g. for the Tau leg, where different bits have same name
    # see: https://github.com/cms-sw/cmssw/blob/b8c94e8ceb9c8060a613aa1f2f5d327db2e8cee5/PhysicsTools/NanoAOD/python/triggerObjects_cff.py#L162C11-L163C64  # noqa
    # or in general the 13_3_X branch of cmssw https://github.com/cms-sw/cmssw/blob/CMSSW_13_3_X/PhysicsTools/NanoAOD/python/triggerObjects_cff.py  # noqa

    # v14
    # from https://github.com/cms-sw/cmssw/tree/f50cf84669608dbe67fd8430660abe651d5b46fd/PhysicsTools/NanoAOD/python/triggerObjects_cff.py  # noqa
    # last update in https://github.com/cms-sw/cmssw/blob/CMSSW_14_0_X/PhysicsTools/NanoAOD/python/triggerObjects_cff.py
    "NanoAODv14": {
        # electron legs
        "Electron": {
            "CaloIdLTrackIdLIsoVL": 1,  # 0
            "WPTightTrackIso": 2,  # 1
            "WPLooseTrackIso": 4,  # 2
            "OverlapFilterPFTau": 8,  # 3
            "DiElectronLeg1": 16,  # 4
            "DiElectronLeg2": 32,  # 5
            "MuEle": 64,  # 6
            "EleTau": 128,  # 7
            "TripleElectron": 256,  # 8
            "SingleMuonDiEle": 512,  # 9
            "DiMuonSingleEle": 1024,  # 10
            "SingleEle_L1DoubleAndSingleEle": 2048,  # 11
            "SingleEle_CaloIdVT_GsfTrkIdT": 4096,  # 12
            "SingleEle_PFJet": 8192,  # 13
            "Photon175_Photon200": 16384,  # 14
            "DoubleEle_CaloIdL_MW_seeded": 32768,  # 15
            "DoubleEle_CaloIdL_MW_unseeded": 65536,  # 16
            "EleTauPNet": 131072,  # 17
        },
        # muon legs
        "Muon": {
            "TrkIsoVVL": 1,  # 0
            "Iso": 2,  # 1
            "OverlapFilterPFTau": 4,  # 2
            "SingleMuon": 8,  # 3
            "DiMuon": 16,  # 4
            "MuEle": 32,  # 5
            "MuTau": 64,  # 6
            "TripleMuon": 128,  # 7
            "DiMuonSingleEle": 256,  # 8
            "SingleMuonDiEle": 512,  # 9
            "Mu50": 1024,  # 10
            "Mu100": 2048,  # 11
            "SingleMuonSinglePhoton": 4096,  # 12
            "MuTauPNet": 8192,  # 13
        },
        # tau legs
        "Tau": {
            "Loose": 1,  # 0
            "Medium": 2,  # 1
            "Tight": 4,  # 2
            "DeepTau": 8,  # 3
            "PNet": 16,  # 4
            "ChargedIso": 32,  # 5
            "Dxy": 64,  # 6
            "ETauFilter": 128,  # 7
            "MuTauFilter": 256,  # 8
            "SingleTau": 512,  # 9
            "VBFDiTau": 1024,  # 10
            "DiTau": 2048,  # 11
            "OverlapFilterIsoEle": 4096,  # 12  # TODO: change name? not overlap filter if PNet, just EleTau
            "OverlapFilterIsoMu": 8192,  # 13  # TODO: change name? not overlap filter if PNet, just MuTau
            "DiTauAndPFJet": 16384,  # 14
            "ETauDisplaced": 32768,  # 15
            "MuTauDisplaced": 65536,  # 16
            "DiTauDisplaced": 131072,  # 17
            "Monitoring": 262144,  # 18
            "MonitoringForVBFIsoTau": 524288,  # 19
            "MonitoringDiTauAndPFJet": 1048576,  # 20
            "MonitoringMuTauDisplaced": 2097152,  # 21
            "OneProng": 4194304,  # 22
            "MonitoringDiTau": 8388608,  # 23
            # TODO: check if MonitoringDiTau (23) will change: filtered are
            # "hltHpsOverlapFilterIsoMu24*DitauWPDeepTauPFTau35Monitoring" -> ditau+muon?
            # and "hltSinglePFJet30PNetTauhTag*WPL2MuXXTauYY" -> single tau from pfjet and something with muons?
            # this does not seem like it is monitoring Ditau at all, but ok... we don't use it anyway
            "OverlapFilter": 16777216,  # 24
            "VBFDoubleTauMonitoring": 33554432,  # 25
            "MonitoringSingleTau": 67108864,  # 26
            "MatchL1HLT": 134217728,  # 27
            "Hps": 268435456,  # 28
            "SinglePFTauFilter": 536870912,  # 29
            "VBFSingleTau": 1073741824,  # 30
            # manually created bit combinations for NanoAOD version compatibility
            "DeeptauDiTau": 2048 + 8,  # 11,3
        },
        # jet legs
        "Jet": {
            "4PixelOnlyPFCentralJetTightIDPt20": 1,  # 0
            "3PixelOnlyPFCentralJetTightIDPt30": 2,  # 1
            "PFJetFilterTwoC30": 4,  # 2
            "4PFCentralJetTightIDPt30": 8,  # 3
            "4PFCentralJetTightIDPt35": 16,  # 4
            "QuadCentralJet30": 32,  # 5
            "2PixelOnlyPFCentralJetTightIDPt40": 64,  # 6
            "L1sTripleJetVBF_orHTT_orDoubleJet_orSingleJet": 128,  # 7
            "3PFCentralJetTightIDPt40": 256,  # 8
            "3PFCentralJetTightIDPt45": 512,  # 9
            "L1sQuadJetsHT": 1024,  # 10
            "BTagCaloDeepCSVp17Double": 2048,  # 11
            "PFCentralJetLooseIDQuad30": 4096,  # 12
            "1PFCentralJetLooseID75": 8192,  # 13
            "2PFCentralJetLooseID60": 16384,  # 14
            "3PFCentralJetLooseID45": 32768,  # 15
            "4PFCentralJetLooseID40": 65536,  # 16
            "DoubleTau+Jet": 131072,  # 17  # now also contains PNet paths
            "VBFcrossCleanedDeepTauPFTau": 262144,  # 18  # now more general VBFDiTauJets  TODO: change name?
            "VBFcrossCleanedUsingDijetCorr": 524288,  # 19  # now more general VBFSingleTauJets  TODO: change name?
            "MonitoringMuon+Tau+Jet": 1048576,  # 20
            "2PFCentralJetTightIDPt50": 2097152,  # 21
            "1PixelOnlyPFCentralJetTightIDPt60": 4194304,  # 22
            "1PFCentralJetTightIDPt70": 8388608,  # 23
            "BTagPFDeepJet1p5Single": 16777216,  # 24
            "BTagPFDeepJet4p5Triple": 33554432,  # 25
            "2BTagSumOR2BTagMeanPaths": 67108864,  # 26
            "2/1PixelOnlyPFCentralJetTightIDPt20/50": 134217728,  # 27
            "2PFCentralJetTightIDPt30": 268435456,  # 28
            "1PFCentralJetTightIDPt60": 536870912,  # 29
            "PF2CentralJetPt30PNet2BTagMean0p50": 1073741824,  # 30
        },
    },
}


# 2016 triggers as per AN of CMS-HIG-20-010 (AN2018_121_v11-1)
def add_triggers_2016(config: od.Config) -> None:
    """
    Adds all triggers to a *config*. For the conversion from filter names to trigger bits, see
    https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py.
    """
    config.x.triggers = od.UniqueObjectIndex(Trigger, [
        #
        # e tauh (NO Triggers in AN)
        # used the triggers from https://twiki.cern.ch/twiki/bin/view/CMS/TauTrigger#Tau_Triggers_in_NanoAOD_2016
        Trigger(
            name="HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_SingleL1",
            id=401,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    # min_pt=26.0,  # TODO
                    # filter names:
                    #
                    trigger_bits=None,  # TODO
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=22.0,  # TODO
                    # filter names:
                    #
                    trigger_bits=None,  # TODO
                ),
            ],
            applies_to_dataset=(
                lambda dataset_inst: dataset_inst.is_mc or (dataset_inst.x.era <= "E")  # TODO: to be checked!
                # does not exist for run F on but should only be used until run 276215 -> which era?
            ),
            tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        ),
        Trigger(
            name="HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau20",
            id=402,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    # min_pt=26.0,  # TODO
                    # filter names:
                    #
                    trigger_bits=None,  # TODO
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=22.0,  # TODO
                    # filter names:
                    #
                    trigger_bits=None,  # TODO
                ),
            ],
            applies_to_dataset=(
                lambda dataset_inst: dataset_inst.is_data and dataset_inst.x.era <= "E"  # TODO: to be checked!
                # does not exist for run F on but should only be used between run 276215 and 278270 -> which eras?
            ),
            tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        ),
        Trigger(
            name="HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau30",
            id=403,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    # min_pt=26.0,  # TODO
                    # filter names:
                    #
                    trigger_bits=None,  # TODO
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=32.0,  # TODO
                    # filter names:
                    #
                    trigger_bits=None,  # TODO
                ),
            ],
            applies_to_dataset=(
                lambda dataset_inst: dataset_inst.is_data and dataset_inst.x.era >= "E"  # TODO: to be checked!
                # does not exist until run E but should only be used after run 278270 -> which era?
            ),
            tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        ),

        #
        # mu tauh
        #
        Trigger(
            name="HLT_IsoMu19_eta2p1_LooseIsoPFTau20",
            id=301,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    # min_pt=22,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=23,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
            ],
            tags={"cross_trigger", "cross_mu_tau", "channel_mu_tau"},
        ),
        Trigger(
            name="HLT_IsoMu19_eta2p1_LooseIsoPFTau20_SingleL1",
            id=302,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    # min_pt=22,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=23,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
            ],
            tags={"cross_trigger", "cross_mu_tau", "channel_mu_tau"},
        ),

        #
        # tauh tauh
        #
        Trigger(
            name="HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Reg",
            id=501,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=38,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=38,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or
                (dataset_inst.x.era >= "B" and dataset_inst.x.era <= "F")
            ),
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),
        Trigger(
            name="HLT_DoubleMediumCombinedIsoPFTau35_Trk1_eta2p1_Reg",
            id=502,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=38,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=38,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "H"),
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),

        #
        # vbf  (NO Triggers)
        #
    ])

    if config.campaign.has_tag("preVFP"):
        #
        # single electron
        #
        config.x.triggers.add(
            name="HLT_Ele25_eta2p1_WPTight_Gsf",
            id=201,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    # min_pt=28,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
            ],
            tags={"single_trigger", "single_e", "channel_e_tau"},
        )
        #
        # single muon
        #
        config.x.triggers.add(
            name="HLT_IsoMu22",
            id=101,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    # min_pt=25,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
            ],
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        )
        config.x.triggers.add(
            name="HLT_IsoMu22_eta2p1",
            id=102,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    # min_pt=25,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
            ],
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        )
        config.x.triggers.add(
            name="HLT_IsoTkMu22",
            id=103,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    # min_pt=25,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
            ],
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        )
        config.x.triggers.add(
            name="HLT_IsoTkMu22_eta2p1",
            id=104,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    # min_pt=25,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
            ],
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        )


def add_triggers_2017(config: od.Config) -> None:
    """
    Adds all triggers to a *config*. For the conversion from filter names to trigger bits, see
    https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py.
    """
    config.x.triggers = od.UniqueObjectIndex(Trigger, [
        #
        # single electron
        #
        Trigger(
            name="HLT_Ele32_WPTight_Gsf",
            id=202,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    # min_pt=35.0,
                    # filter names:
                    # hltEle32WPTightGsfTrackIsoFilter
                    trigger_bits=2,
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "D"),
            tags={"single_trigger", "single_e", "channel_e_tau"},
        ),
        Trigger(
            name="HLT_Ele32_WPTight_Gsf_L1DoubleEG",
            id=203,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    # min_pt=35.0,
                    # filter names:
                    # hltEle32L1DoubleEGWPTightGsfTrackIsoFilter
                    # hltEGL1SingleEGOrFilter
                    trigger_bits=2 + 1024,
                ),
            ],
            tags={"single_trigger", "single_e", "channel_e_tau"},
        ),
        Trigger(
            name="HLT_Ele35_WPTight_Gsf",
            id=204,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    # min_pt=38.0,
                    # filter names:
                    # hltEle35noerWPTightGsfTrackIsoFilter
                    trigger_bits=2,
                ),
            ],
            tags={"single_trigger", "single_e", "channel_e_tau"},
        ),

        #
        # single muon
        #
        Trigger(
            name="HLT_IsoMu24",
            id=105,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    # min_pt=26.0,
                    # filter names:
                    # hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p07
                    trigger_bits=2,
                ),
            ],
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        ),
        Trigger(
            name="HLT_IsoMu27",
            id=106,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    # min_pt=29.0,
                    # filter names:
                    # hltL3crIsoL1sMu22Or25L1f0L2f10QL3f27QL3trkIsoFiltered0p07
                    trigger_bits=2,
                ),
            ],
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        ),

        #
        # e tauh
        #
        Trigger(
            name="HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTau30_eta2p1_CrossL1",
            id=404,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    # min_pt=27.0,
                    # filter names:
                    # hltEle24erWPTightGsfTrackIsoFilterForTau
                    # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
                    trigger_bits=2 + 64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=35.0,
                    # filter names:
                    # hltSelectedPFTau30LooseChargedIsolationL1HLTMatched
                    # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
                    trigger_bits=1024 + 256,
                ),
            ],
            tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        ),

        #
        # mu tauh
        #
        Trigger(
            name="HLT_IsoMu20_eta2p1_LooseChargedIsoPFTau27_eta2p1_CrossL1",
            id=303,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    # min_pt=22.0,
                    # filter names:
                    # hltL3crIsoL1sMu18erTau24erIorMu20erTau24erL1f0L2f10QL3f20QL3trkIsoFiltered0p07
                    # hltOverlapFilterIsoMu20LooseChargedIsoPFTau27L1Seeded
                    trigger_bits=2 + 64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=32.0,
                    # filter names:
                    # hltSelectedPFTau27LooseChargedIsolationAgainstMuonL1HLTMatched or
                    # hltOverlapFilterIsoMu20LooseChargedIsoPFTau27L1Seeded
                    trigger_bits=1024 + 512,
                ),
            ],
            tags={"cross_trigger", "cross_mu_tau", "channel_mu_tau"},
        ),

        #
        # tauh tauh
        #
        Trigger(
            name="HLT_DoubleMediumChargedIsoPFTau35_Trk1_eta2p1_Reg",
            id=503,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=40.0,
                    # filter names:
                    # hltDoublePFTau35TrackPt1MediumChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=40.0,
                    # filter names:
                    # hltDoublePFTau35TrackPt1MediumChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
            ],
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),
        Trigger(
            name="HLT_DoubleTightChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg",
            id=504,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=40.0,
                    # filter names:
                    # hltDoublePFTau35TrackPt1TightChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=40.0,
                    # filter names:
                    # hltDoublePFTau35TrackPt1TightChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data),
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),
        Trigger(
            name="HLT_DoubleMediumChargedIsoPFTau40_Trk1_TightID_eta2p1_Reg",
            id=505,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=45.0,
                    # filter names:
                    # hltDoublePFTau40TrackPt1MediumChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=45.0,
                    # filter names:
                    # hltDoublePFTau40TrackPt1MediumChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data),
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),
        Trigger(
            name="HLT_DoubleTightChargedIsoPFTau40_Trk1_eta2p1_Reg",
            id=506,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=45.0,
                    # filter names:
                    # hltDoublePFTau40TrackPt1TightChargedIsolationDz02Reg
                    trigger_bits=64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=45.0,
                    # filter names:
                    # hltDoublePFTau40TrackPt1TightChargedIsolationDz02Reg
                    trigger_bits=64,
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data),
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),

        #
        # vbf
        #
        Trigger(
            name="HLT_VBF_DoubleLooseChargedIsoPFTau20_Trk1_eta2p1_Reg",
            id=601,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=25.0,
                    # filter names:
                    # hltDoublePFTau20TrackPt1LooseChargedIsolation
                    trigger_bits=2048,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=25.0,
                    # filter names:
                    # hltDoublePFTau20TrackPt1LooseChargedIsolation
                    trigger_bits=2048,
                ),
                # additional leg infos for vbf jets
                # TODO check if vbf legs are needed
                TriggerLeg(
                    # min_pt=115.0,
                    # filter names:
                    # hltMatchedVBFOnePFJet2CrossCleanedFromDoubleLooseChargedIsoPFTau20
                    trigger_bits=1,
                ),
                TriggerLeg(
                    # min_pt=40.0,
                    # filter names:
                    # hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleLooseChargedIsoPFTau20
                    trigger_bits=1,
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "D"),
            tags={"cross_trigger", "cross_tau_tau_vbf", "channel_tau_tau"},
        ),
    ])


def add_triggers_2018(config: od.Config) -> None:
    config.x.triggers = od.UniqueObjectIndex(Trigger, [
        #
        # single electron
        #
        Trigger(
            name="HLT_Ele32_WPTight_Gsf",
            id=202,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    # min_pt=35.0,
                    # filter names:
                    # hltEle32WPTightGsfTrackIsoFilter
                    trigger_bits=2,
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "D"),
            tags={"single_trigger", "single_e", "channel_e_tau"},
        ),
        Trigger(
            name="HLT_Ele35_WPTight_Gsf",
            id=204,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    # min_pt=38.0,
                    # filter names:
                    # hltEle35noerWPTightGsfTrackIsoFilter
                    trigger_bits=2,
                ),
            ],
            tags={"single_trigger", "single_e", "channel_e_tau"},
        ),

        #
        # single muon
        #
        Trigger(
            name="HLT_IsoMu24",
            id=105,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    # min_pt=26.0,
                    # filter names:
                    # hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p07
                    trigger_bits=2,
                ),
            ],
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        ),
        Trigger(
            name="HLT_IsoMu27",
            id=106,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    # min_pt=29.0,
                    # filter names:
                    # hltL3crIsoL1sMu22Or25L1f0L2f10QL3f27QL3trkIsoFiltered0p07
                    trigger_bits=2,
                ),
            ],
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        ),

        #
        # e tauh
        #
        Trigger(
            name="HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTau30_eta2p1_CrossL1",
            id=404,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    # min_pt=27.0,
                    # filter names:
                    # hltEle24erWPTightGsfTrackIsoFilterForTau
                    # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
                    trigger_bits=2 + 64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=35.0,
                    # filter names:
                    # hltSelectedPFTau30LooseChargedIsolationL1HLTMatched
                    # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
                    trigger_bits=1024 + 256,
                ),
            ],
            # the non-HPS path existed only for data and is fully covered in MC below
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data),
            tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        ),

        #
        # mu tauh
        #
        Trigger(
            name="HLT_IsoMu20_eta2p1_LooseChargedIsoPFTau27_eta2p1_CrossL1",
            id=303,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    # min_pt=22.0,
                    # filter names:
                    # hltL3crIsoL1sMu18erTau24erIorMu20erTau24erL1f0L2f10QL3f20QL3trkIsoFiltered0p07
                    # hltOverlapFilterIsoMu20LooseChargedIsoPFTau27L1Seeded
                    trigger_bits=2 + 64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=32.0,
                    # filter names:
                    # hltSelectedPFTau27LooseChargedIsolationAgainstMuonL1HLTMatched or
                    # hltOverlapFilterIsoMu20LooseChargedIsoPFTau27L1Seeded
                    trigger_bits=1024 + 512,
                ),
            ],
            # the non-HPS path existed only for data and is fully covered in MC below
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data),
            tags={"cross_trigger", "cross_mu_tau", "channel_mu_tau"},
        ),

        #
        # tauh tauh
        #
        Trigger(
            name="HLT_DoubleTightChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg",
            id=504,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=40.0,
                    # filter names:
                    # hltDoublePFTau35TrackPt1TightChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=40.0,
                    # filter names:
                    # hltDoublePFTau35TrackPt1TightChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data),
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),
        Trigger(
            name="HLT_DoubleMediumChargedIsoPFTau40_Trk1_TightID_eta2p1_Reg",
            id=505,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=45.0,
                    # filter names:
                    # hltDoublePFTau40TrackPt1MediumChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=45.0,
                    # filter names:
                    # hltDoublePFTau40TrackPt1MediumChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data),
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),
        Trigger(
            name="HLT_DoubleTightChargedIsoPFTau40_Trk1_eta2p1_Reg",
            id=506,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=45.0,
                    # filter names:
                    # hltDoublePFTau40TrackPt1TightChargedIsolationDz02Reg
                    trigger_bits=64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=45.0,
                    # filter names:
                    # hltDoublePFTau40TrackPt1TightChargedIsolationDz02Reg
                    trigger_bits=64,
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data),
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),

        #
        # vbf
        #
        Trigger(
            name="HLT_VBF_DoubleLooseChargedIsoPFTau20_Trk1_eta2p1_Reg",
            id=601,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=25.0,
                    # filter names:
                    # hltDoublePFTau20TrackPt1LooseChargedIsolation
                    trigger_bits=2048,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=25.0,
                    # filter names:
                    # hltDoublePFTau20TrackPt1LooseChargedIsolation
                    trigger_bits=2048,
                ),
                # additional leg infos for vbf jets
                TriggerLeg(
                    # min_pt=115.0,
                    # filter names:
                    # hltMatchedVBFOnePFJet2CrossCleanedFromDoubleLooseChargedIsoPFTau20
                    trigger_bits=1,
                ),
                TriggerLeg(
                    # min_pt=40.0,
                    # filter names:
                    # hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleLooseChargedIsoPFTau20
                    trigger_bits=1,
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "D"),
            tags={"cross_trigger", "cross_tau_tau_vbf", "channel_tau_tau"},
        ),
    ])


def add_triggers_2022(config: od.Config) -> None:
    """
    Adds all triggers to a *config*. For the conversion from filter names to trigger bits, see
    https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py.
    Tau Trigger: https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauTrigger#Trigger_Table_for_2022
    Electron Trigger: https://twiki.cern.ch/twiki/bin/view/CMS/EgHLTRunIIISummary
    Muon Trigger: https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLT2022
    """
    config.x.triggers = od.UniqueObjectIndex(Trigger, [
        #
        # single electron
        #
        Trigger(
            name="HLT_Ele30_WPTight_Gsf",
            id=205,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    # min_pt=31.0,
                    # filter names:
                    # hltEle30WPTightGsfTrackIsoFilter (WPTightTrackIso)
                    trigger_bits=2,
                ),
            ],
            applies_to_dataset=(
                lambda dataset_inst: dataset_inst.is_mc or
                dataset_inst.has_tag("etau") or
                dataset_inst.has_tag("emu_from_e") or
                dataset_inst.has_tag("emu_from_mu")
            ),
            tags={"single_trigger", "single_e", "channel_e_tau"},
        ),

        #
        # single muon
        #
        Trigger(
            name="HLT_IsoMu24",
            id=105,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    # min_pt=26.0,
                    # filter names:
                    # hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p08 (1mu + Iso)
                    trigger_bits=2 + 8,
                ),
            ],
            applies_to_dataset=(
                lambda dataset_inst: dataset_inst.is_mc or
                dataset_inst.has_tag("mutau") or
                dataset_inst.has_tag("emu_from_e") or
                dataset_inst.has_tag("emu_from_mu") or
                dataset_inst.has_tag("mumu")
            ),
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        ),

        #
        # e tauh
        #
        Trigger(
            name="HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
            id=405,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    # min_pt=25.0,
                    # filter names:
                    # hltHpsOverlapFilterIsoEle24WPTightGsfLooseETauWPDeepTauPFTau30 (DeepTau + OverlapFilter)
                    trigger_bits=8 + 64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=35.0,
                    # filter names:
                    # (DeepTau + HPS + Overlap)
                    # hltHpsOverlapFilterIsoEle24WPTightGsfLooseETauWPDeepTauPFTau30
                    trigger_bits=8 + 32 + 256,
                ),
            ],
            applies_to_dataset=(
                lambda dataset_inst: dataset_inst.is_mc or
                dataset_inst.has_tag("etau")
            ),
            tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        ),

        #
        # mu tauh
        #
        Trigger(
            name="HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1",
            id=304,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    # min_pt=22.0,
                    # filter names:
                    # hltHpsOverlapFilterIsoMu20LooseMuTauWPDeepTauPFTau27L1Seeded (OverlapFilter PFTau)
                    trigger_bits=4 + 64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=32.0,
                    # filter names:
                    # (DeepTau + HPS + Overlap + L1Seeded)
                    # hltHpsOverlapFilterIsoMu20LooseMuTauWPDeepTauPFTau27L1Seeded
                    trigger_bits=8 + 32 + 512 + 262144,
                ),
            ],
            applies_to_dataset=(
                lambda dataset_inst: dataset_inst.is_mc or
                dataset_inst.has_tag("mutau")
            ),
            tags={"cross_trigger", "cross_mu_tau", "channel_mu_tau"},
        ),

        #
        # tauh tauh
        #
        Trigger(
            name="HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1",
            id=507,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=40.0,
                    # filter names:
                    # hltHpsDoublePFTau35MediumDitauWPDeepTauL1HLTMatched (Deeptau + HPS + DeepTauDiTau)
                    trigger_bits=8 + 32 + 128,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=40.0,
                    # filter names:
                    # hltHpsDoublePFTau35MediumDitauWPDeepTauL1HLTMatched (Deeptau + HPS + DeepTauDiTau)
                    trigger_bits=8 + 32 + 128,
                ),
            ],
            applies_to_dataset=(
                lambda dataset_inst: dataset_inst.is_mc or
                dataset_inst.has_tag("tautau")
            ),
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),

        #
        # vbf
        #
        Trigger(
            name="HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1",
            id=602,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=25.0,
                    # filter names:
                    # (DeepTau + HPS + run 3 VBF+ditau)
                    # hltHpsDoublePFTau20TrackDeepTauDitauWPForVBFIsoTau
                    trigger_bits=8 + 32 + 4096,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=25.0,
                    # filter names:
                    # hltHpsDoublePFTau20TrackDeepTauDitauWPForVBFIsoTau
                    trigger_bits=8 + 32 + 4096,
                ),
                # additional leg infos for vbf jets
                TriggerLeg(
                    pdg_id=1,
                    # min_pt=115.0,
                    # filter names:
                    # The filters are applied to the lepton
                    # Taking the loosest filter for the Jets with the pt cut

                    # maybe hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20?
                    # (VBF cross-cleaned from medium deeptau PFTau)
                    # trigger_bits=262144,  # does not work in v12 and v13  # TODO: add it for v14
                ),
                TriggerLeg(
                    pdg_id=1,
                    # min_pt=40.0,
                    # filter names:
                    # The filters are applied to the lepton

                    # maybe hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20?
                    # (VBF cross-cleaned from medium deeptau PFTau)
                    # trigger_bits=262144,  # does not work in v12 and v13  # TODO: add it for v14
                ),
            ],
            applies_to_dataset=(
                lambda dataset_inst: dataset_inst.is_mc or
                dataset_inst.has_tag("tautau")
            ),
            tags={"cross_trigger", "cross_tau_tau_vbf", "channel_tau_tau"},
        ),

        # Currently disabled since it may not be needed
        # Trigger(
        #     name="HLT_DoublePFJets40_Mass500_MediumDeepTauPFTauHPS45_L2NN_MediumDeepTauPFTauHPS20_eta2p1",
        #     id=604,
        #     legs=[
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
        Trigger(
            name="HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60",
            id=701,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=35.0,
                    # filter names:
                    # (DeepTau + Hps + ditau+PFJet)
                    # hltHpsOverlapFilterDeepTauDoublePFTau30PFJet60
                    trigger_bits=8 + 32 + 16384,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=35.0,
                    # filter names:
                    # hltHpsOverlapFilterDeepTauDoublePFTau30PFJet60
                    trigger_bits=8 + 32 + 16384,
                ),
                TriggerLeg(
                    pdg_id=1,
                    # min_pt=65.0,
                    # filter names:
                    # Filters are applied to the leptons
                    # Taking the loosest filter for the Jets with the pt cut

                    # hltHpsOverlapFilterDeepTauDoublePFTau30PFJet60
                    # (DoubleTau + Jet) -> 17
                    trigger_bits=131072,
                ),
            ],
            applies_to_dataset=(
                lambda dataset_inst: dataset_inst.is_mc or
                dataset_inst.has_tag("tautau")
            ),
            tags={"cross_trigger", "cross_tau_tau_jet", "channel_tau_tau"},
        ),
    ])


def add_triggers_2023(config: od.Config) -> None:
    """
    Adds all triggers to a *config*. For the conversion from filter names to trigger bits, see
    https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py.
    """
    config.x.triggers = od.UniqueObjectIndex(Trigger, [
        #
        # single electron
        #
        Trigger(
            name="HLT_Ele30_WPTight_Gsf",
            id=205,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    # min_pt=31.0,
                    # filter names:
                    # WPTightTrackIso
                    trigger_bits=2,
                ),
            ],
            applies_to_dataset=(
                lambda dataset_inst: dataset_inst.is_mc or
                dataset_inst.has_tag("etau") or
                dataset_inst.has_tag("emu_from_e") or
                dataset_inst.has_tag("emu_from_mu")
            ),
            tags={"single_trigger", "single_e", "channel_e_tau"},
        ),

        #
        # single muon
        #
        Trigger(
            name="HLT_IsoMu24",
            id=105,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    # min_pt=25.0,
                    # filter names:
                    # hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p08 (1mu + Iso)
                    trigger_bits=2 + 8,
                ),
            ],
            applies_to_dataset=(
                lambda dataset_inst: dataset_inst.is_mc or
                dataset_inst.has_tag("mutau") or
                dataset_inst.has_tag("emu_from_e") or
                dataset_inst.has_tag("emu_from_mu") or
                dataset_inst.has_tag("mumu")
            ),
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        ),

        #
        # e tauh
        #
        Trigger(
            name="HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
            id=405,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    # min_pt=25.0,
                    # filter names:
                    # hltEle24erWPTightGsfTrackIsoFilterForTau
                    # hltHpsOverlapFilterIsoEle24WPTightGsfLooseETauWPDeepTauPFTau30 (OverlapFilter)
                    trigger_bits=8,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=35.0,
                    # filter names:
                    # hltHpsSelectedPFTau30LooseETauWPDeepTauL1HLTMatched (DeepTau + HPS)
                    # hltHpsOverlapFilterIsoEle24WPTightGsfLooseETauWPDeepTauPFTau30
                    trigger_bits=8 + 32,
                ),
            ],
            applies_to_dataset=(
                lambda dataset_inst: dataset_inst.is_mc or
                dataset_inst.has_tag("etau")
            ),
            tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        ),

        #
        # mu tauh
        #
        Trigger(
            name="HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1",
            id=304,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    # min_pt=21.0,
                    # filter names:
                    # hltL3crIsoBigORMu18erTauXXer2p1L1f0L2f10QL3f20QL3trkIsoFiltered0p08
                    # hltHpsOverlapFilterIsoMu20LooseMuTauWPDeepTauPFTau27L1Seeded (OverlapFilter PFTau)
                    trigger_bits=4,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=32.0,
                    # filter names:
                    # hltHpsSelectedPFTau27LooseMuTauWPDeepTauVsJetsAgainstMuonL1HLTMatched (DeepTau + HPS)
                    # hltHpsOverlapFilterIsoMu20LooseMuTauWPDeepTauPFTau27L1Seeded
                    trigger_bits=8 + 32,
                ),
            ],
            applies_to_dataset=(
                lambda dataset_inst: dataset_inst.is_mc or
                dataset_inst.has_tag("mutau")
            ),
            tags={"cross_trigger", "cross_mu_tau", "channel_mu_tau"},
        ),

        #
        # tauh tauh
        #
        Trigger(
            name="HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1",
            id=507,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=40.0,
                    # filter names:
                    # hltHpsDoublePFTau35MediumDitauWPDeepTauL1HLTMatched (Deeptau + HPS)
                    trigger_bits=8 + 32,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=40.0,
                    # filter names:
                    # hltHpsDoublePFTau35MediumDitauWPDeepTauL1HLTMatched (Deeptau + HPS)
                    trigger_bits=8 + 32,
                ),
            ],
            applies_to_dataset=(
                lambda dataset_inst: dataset_inst.is_mc or
                dataset_inst.has_tag("tautau")
            ),
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),

        #
        # vbf
        #
        Trigger(
            name="HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1",
            id=602,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=25.0,
                    # filter names:
                    # (DeepTau + HPS + run 3 VBF+ditau)
                    # hltHpsDoublePFTau20TrackDeepTauDitauWPForVBFIsoTau
                    # hltMatchedVBFOnePFJet2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20
                    # hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20
                    trigger_bits=8 + 32 + 4096,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=25.0,
                    # filter names:
                    # hltHpsDoublePFTau20TrackDeepTauDitauWPForVBFIsoTau
                    # hltMatchedVBFOnePFJet2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20
                    # hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20
                    trigger_bits=8 + 32 + 4096,
                ),
            ],
            applies_to_dataset=(
                lambda dataset_inst: dataset_inst.is_mc or
                dataset_inst.has_tag("tautau")
            ),
            tags={"cross_trigger", "cross_tau_tau_vbf", "channel_tau_tau"},
        ),

        #
        # tau tau jet
        #
        Trigger(
            name="HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60",
            id=701,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=35.0,
                    # filter names:
                    # (TightOOSCPhotons + di-tau + PFJet)
                    # hltHpsDoublePFTau30MediumDitauWPDeepTauL1HLTMatchedDoubleTauJet
                    trigger_bits=16 + 16384,
                ),
                TriggerLeg(
                    pdg_id=15,
                    # min_pt=35.0,
                    # filter names:
                    # hltHpsDoublePFTau30MediumDitauWPDeepTauL1HLTMatchedDoubleTauJet
                    trigger_bits=16 + 16384,
                ),
            ],
            applies_to_dataset=(
                lambda dataset_inst: dataset_inst.is_mc or
                dataset_inst.has_tag("tautau")
            ),
            tags={"cross_trigger", "cross_tau_tau_jet", "channel_tau_tau"},
        ),
    ])
