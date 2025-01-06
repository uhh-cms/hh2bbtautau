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

701 -> HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60
702 -> HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet75
"""

from dataclasses import dataclass

import order as od

from columnflow.util import DotDict
from columnflow.types import ClassVar

from hbt.config.util import Trigger, TriggerLeg


@dataclass
class Bits:
    v12: int | None = None
    v14: int | None = None

    supported_versions = ClassVar({12, 14})

    def get(self, nano_version: int) -> int:
        if nano_version not in self.supported_versions:
            raise ValueError(f"nano_version {nano_version} not supported")
        return getattr(self, f"v{nano_version}")


# # use the CCLub names for the trigger bits and improve them when necessary
# trigger_bits = DotDict.wrap({
#     # checked with https://github.com/cms-sw/cmssw/blob/CMSSW_13_0_X/PhysicsTools/NanoAOD/python/triggerObjects_cff.py
#     # and in particular https://github.com/cms-sw/cmssw/blob/2defd844e96613d2438b690d10b79c773e02ab57/PhysicsTools/NanoAOD/python/triggerObjects_cff.py  # noqa
#     "e": {
#         "CaloIdLTrackIdLIsoVL": Bits(v12=1),
#         "CaloIdLTrackIdLIsoVVVL": Bits(v14=1),
#         "WPTightTrackIso": 2,
#         "WPLooseTrackIso": 4,
#         "OverlapFilterPFTau": 8,
#         "DiElectron": 16,
#         "MuEle": 32,
#         "EleTau": 64,
#         "TripleElectron": 128,
#         "SingleMuonDiEle": 256,
#         "DiMuonSingleEle": 512,
#         "SingleEle_L1DoubleAndSingleEle": 1024,
#         "SingleEle_CaloIdVT_GsfTrkIdT": 2048,
#         "SingleEle_PFJet": 4096,
#         "Photon175_Photon200": 8192,
#     },
# })


def get_bit_sum(nano_version: int, obj_name: str, names: list[str | None]) -> int:
    return sum(
        trigger_bits[obj_name][name].get(nano_version)
        for name in names
        if name is not None
    )


# use the CCLub names for the trigger bits and improve them when necessary
trigger_bits = DotDict.wrap({
    # checked with https://github.com/cms-sw/cmssw/blob/CMSSW_13_0_X/PhysicsTools/NanoAOD/python/triggerObjects_cff.py
    # and in particular https://github.com/cms-sw/cmssw/blob/2defd844e96613d2438b690d10b79c773e02ab57/PhysicsTools/NanoAOD/python/triggerObjects_cff.py  # noqa
    12: {
        "e": {
            "CaloIdLTrackIdLIsoVL": 1,
            "WPTightTrackIso": 2,
            "WPLooseTrackIso": 4,
            "OverlapFilterPFTau": 8,
            "DiElectron": 16,
            "MuEle": 32,
            "EleTau": 64,
            "TripleElectron": 128,
            "SingleMuonDiEle": 256,
            "DiMuonSingleEle": 512,
            "SingleEle_L1DoubleAndSingleEle": 1024,
            "SingleEle_CaloIdVT_GsfTrkIdT": 2048,
            "SingleEle_PFJet": 4096,
            "Photon175_Photon200": 8192,
        },
        "mu": {
            "TrkIsoVVL": 1,
            "Iso": 2,
            "OverlapFilterPFTau": 4,
            "SingleMuon": 8,
            "DiMuon": 16,
            "MuEle": 32,
            "MuTau": 64,
            "TripleMuon": 128,
            "DiMuonSingleEle": 256,
            "SingleMuonDiEle": 512,
            "Mu50": 1024,
            "Mu100": 2048,
            "SingleMuonSinglePhoton": 4096,
        },
        "tau": {
            "LooseChargedIso": 1,
            "MediumChargedIso": 2,
            "TightChargedIso": 4,
            "DeepTau": 8,
            "TightOOSCPhotons": 16,
            "Hps": 32,
            "ChargedIsoDiTau": 64,
            "DeeptauDiTau": 128,
            "OverlapFilterIsoEle": 256,
            "OverlapFilterIsoMu": 512,
            "SingleTauOrTauMet": 1024,
            "VBFpDoublePFTau_run2": 2048,
            "VBFpDoublePFTau_run3": 4096,
            "DiPFJetAndDiTau": 8192,
            "DiTauAndPFJet": 16384,
            "DisplacedTau": 32768,
            "Monitoring": 65536,
            "RegionalPaths": 131072,
            "L1SeededPaths": 262144,
            # "MatchL1HLT": 262144,  # alias for v14 compatibility
            "1Prong": 524288,
        },
        "jet": {
            "4PixelOnlyPFCentralJetTightIDPt20": 1,
            "3PixelOnlyPFCentralJetTightIDPt30": 2,
            "PFJetFilterTwoC30": 4,
            "4PFCentralJetTightIDPt30": 8,
            "4PFCentralJetTightIDPt35": 16,
            "QuadCentralJet30": 32,
            "2PixelOnlyPFCentralJetTightIDPt40": 64,
            "L1sTripleJetVBF_orHTT_orDoubleJet_orSingleJet": 128,
            "3PFCentralJetTightIDPt40": 256,
            "3PFCentralJetTightIDPt45": 512,
            "L1sQuadJetsHT": 1024,
            "BTagCaloDeepCSVp17Double": 2048,
            "PFCentralJetLooseIDQuad30": 4096,
            "1PFCentralJetLooseID75": 8192,
            "2PFCentralJetLooseID60": 16384,
            "3PFCentralJetLooseID45": 32768,
            "4PFCentralJetLooseID40": 65536,
            "DoubleTau+Jet": 131072,
            "VBFcrossCleanedDeepTauPFTau": 262144,  # TODO: change name? idea comes from v14
            "VBFcrossCleanedUsingDijetCorr": 524288,  # TODO: change name? idea comes from v14
            "MonitoringMuon+Tau+Jet": 1048576,
            "2PFCentralJetTightIDPt50": 2097152,
            "1PixelOnlyPFCentralJetTightIDPt60": 4194304,
            "1PFCentralJetTightIDPt70": 8388608,
            "BTagPFDeepJet1p5Single": 16777216,
            "BTagPFDeepJet4p5Triple": 33554432,
            "2BTagSumOR2BTagMeanPaths": 67108864,
            "2/1PixelOnlyPFCentralJetTightIDPt20/50": 134217728,
            "2PFCentralJetTightIDPt30": 268435456,
            "1PFCentralJetTightIDPt60": 536870912,
            "PF2CentralJetPt30PNet2BTagMean0p50": 1073741824,
        },
    },

    # from https://github.com/cms-sw/cmssw/tree/f50cf84669608dbe67fd8430660abe651d5b46fd/PhysicsTools/NanoAOD/python/triggerObjects_cff.py  # noqa
    # last update in https://github.com/cms-sw/cmssw/blob/CMSSW_14_0_X/PhysicsTools/NanoAOD/python/triggerObjects_cff.py
    14: {
        "e": {
            "CaloIdLTrackIdLIsoVL": 1,
            "WPTightTrackIso": 2,
            "WPLooseTrackIso": 4,
            "OverlapFilterPFTau": 8,
            "DiElectronLeg1": 16,
            "DiElectronLeg2": 32,
            "MuEle": 64,
            "EleTau": 128,
            "TripleElectron": 256,
            "SingleMuonDiEle": 512,
            "DiMuonSingleEle": 1024,
            "SingleEle_L1DoubleAndSingleEle": 2048,
            "SingleEle_CaloIdVT_GsfTrkIdT": 4096,
            "SingleEle_PFJet": 8192,
            "Photon175_Photon200": 16384,
            "DoubleEle_CaloIdL_MW_seeded": 32768,
            "DoubleEle_CaloIdL_MW_unseeded": 65536,
            "EleTauPNet": 131072,
        },
        "mu": {
            "TrkIsoVVL": 1,
            "Iso": 2,
            "OverlapFilterPFTau": 4,
            "SingleMuon": 8,
            "DiMuon": 16,
            "MuEle": 32,
            "MuTau": 64,
            "TripleMuon": 128,
            "DiMuonSingleEle": 256,
            "SingleMuonDiEle": 512,
            "Mu50": 1024,
            "Mu100": 2048,
            "SingleMuonSinglePhoton": 4096,
            "MuTauPNet": 8192,
        },
        "tau": {
            "Loose": 1,
            "Medium": 2,
            "Tight": 4,
            "DeepTau": 8,
            "PNet": 16,
            "ChargedIso": 32,
            "Dxy": 64,
            "ETauFilter": 128,
            "MuTauFilter": 256,
            "SingleTau": 512,
            "VBFDiTau": 1024,
            "DiTau": 2048,
            "OverlapFilterIsoEle": 4096,  # TODO: change name? not overlap filter if PNet, just EleTau
            "OverlapFilterIsoMu": 8192,  # TODO: change name? not overlap filter if PNet, just MuTau
            "DiTauAndPFJet": 16384,
            "ETauDisplaced": 32768,
            "MuTauDisplaced": 65536,
            "DiTauDisplaced": 131072,
            "Monitoring": 262144,
            "MonitoringForVBFIsoTau": 524288,
            "MonitoringDiTauAndPFJet": 1048576,
            "MonitoringMuTauDisplaced": 2097152,
            "OneProng": 4194304,
            "MonitoringDiTau": 8388608,  # surprising filters, but we don't use it anyway
            "OverlapFilter": 16777216,
            "VBFDoubleTauMonitoring": 33554432,
            "MonitoringSingleTau": 67108864,
            "MatchL1HLT": 134217728,
            "Hps": 268435456,
            "SinglePFTauFilter": 536870912,
            "VBFSingleTau": 1073741824,
            # manually created bit combinations for NanoAOD version compatibility
            "DeeptauDiTau": 2048 + 8,
        },
        "jet": {
            "4PixelOnlyPFCentralJetTightIDPt20": 1,
            "3PixelOnlyPFCentralJetTightIDPt30": 2,
            "PFJetFilterTwoC30": 4,
            "4PFCentralJetTightIDPt30": 8,
            "4PFCentralJetTightIDPt35": 16,
            "QuadCentralJet30": 32,
            "2PixelOnlyPFCentralJetTightIDPt40": 64,
            "L1sTripleJetVBF_orHTT_orDoubleJet_orSingleJet": 128,
            "3PFCentralJetTightIDPt40": 256,
            "3PFCentralJetTightIDPt45": 512,
            "L1sQuadJetsHT": 1024,
            "BTagCaloDeepCSVp17Double": 2048,
            "PFCentralJetLooseIDQuad30": 4096,
            "1PFCentralJetLooseID75": 8192,
            "2PFCentralJetLooseID60": 16384,
            "3PFCentralJetLooseID45": 32768,
            "4PFCentralJetLooseID40": 65536,
            "DoubleTau+Jet": 131072,  # now also contains PNet paths
            "VBFcrossCleanedDeepTauPFTau": 262144,  # now more general VBFDiTauJets  TODO: change name?
            "VBFcrossCleanedUsingDijetCorr": 524288,  # now more general VBFSingleTauJets  TODO: change name?
            "MonitoringMuon+Tau+Jet": 1048576,
            "2PFCentralJetTightIDPt50": 2097152,
            "1PixelOnlyPFCentralJetTightIDPt60": 4194304,
            "1PFCentralJetTightIDPt70": 8388608,
            "BTagPFDeepJet1p5Single": 16777216,
            "BTagPFDeepJet4p5Triple": 33554432,
            "2BTagSumOR2BTagMeanPaths": 67108864,
            "2/1PixelOnlyPFCentralJetTightIDPt20/50": 134217728,
            "2PFCentralJetTightIDPt30": 268435456,
            "1PFCentralJetTightIDPt60": 536870912,
            "PF2CentralJetPt30PNet2BTagMean0p50": 1073741824,
        },
    },
})


# 2016 triggers as per AN of CMS-HIG-20-010 (AN2018_121_v11-1)
def add_triggers_2016(config: od.Config) -> None:
    """
    Adds all triggers to a *config*. For the conversion from filter names to trigger bits, see
    https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py.
    """
    config.x.triggers = od.UniqueObjectIndex(Trigger, [
        #
        # e tauh
        #
        # from https://twiki.cern.ch/twiki/bin/view/CMS/TauTrigger#Tau_Triggers_in_NanoAOD_2016
        Trigger(
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
            tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        ),
        Trigger(
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
            tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        ),
        Trigger(
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
            tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        ),

        #
        # mu tauh
        #
        Trigger(
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
            tags={"cross_trigger", "cross_mu_tau", "channel_mu_tau"},
        ),
        Trigger(
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
            tags={"cross_trigger", "cross_mu_tau", "channel_mu_tau"},
        ),

        #
        # tauh tauh
        #
        Trigger(
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
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),
        Trigger(
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
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),

        #
        # vbf
        #
        # none
    ])

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
            tags={"single_trigger", "single_e", "channel_e_tau"},
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
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
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
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
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
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
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
            tags={"single_trigger", "single_e", "channel_e_tau"},
        ),
        Trigger(
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
            tags={"single_trigger", "single_e", "channel_e_tau"},
        ),
        Trigger(
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
            tags={"single_trigger", "single_e", "channel_e_tau"},
        ),

        #
        # single muon
        #
        Trigger(
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
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        ),
        Trigger(
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
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        ),

        #
        # e tauh
        #
        Trigger(
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
            tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        ),

        #
        # mu tauh
        #
        Trigger(
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
            tags={"cross_trigger", "cross_mu_tau", "channel_mu_tau"},
        ),

        #
        # tauh tauh
        #
        Trigger(
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
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),
        Trigger(
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
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),
        Trigger(
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
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),
        Trigger(
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
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),

        #
        # vbf
        #
        Trigger(
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
                # TODO check if vbf legs are needed
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
            tags={"single_trigger", "single_e", "channel_e_tau"},
        ),
        Trigger(
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
            tags={"single_trigger", "single_e", "channel_e_tau"},
        ),

        #
        # single muon
        #
        Trigger(
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
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        ),
        Trigger(
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
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        ),

        #
        # e tauh
        #
        Trigger(
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
            tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        ),

        #
        # mu tauh
        #
        Trigger(
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
            tags={"cross_trigger", "cross_mu_tau", "channel_mu_tau"},
        ),

        #
        # tauh tauh
        #
        Trigger(
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
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),
        Trigger(
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
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),
        Trigger(
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
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),

        #
        # vbf
        #
        Trigger(
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
    # get trigger bits for the requested nano version
    # bits = trigger_bits[config.campaign.x.version]

    config.x.triggers = od.UniqueObjectIndex(Trigger, [
        #
        # single electron
        #
        Trigger(
            name="HLT_Ele30_WPTight_Gsf",
            id=205,
            legs=dict(
                e=TriggerLeg(
                    pdg_id=11,
                    # min_pt=31.0,
                    # filter names:
                    # hltEle30WPTightGsfTrackIsoFilter (WPTightTrackIso)
                    # Tight works too for v14, but redundant
                    # trigger_bits=WPTightTrackIso,
                    trigger_bits=2,
                ),
            ),
            applies_to_dataset=(lambda dataset_inst: (
                dataset_inst.is_mc or
                dataset_inst.has_tag("etau") or
                dataset_inst.has_tag("emu_from_e") or
                dataset_inst.has_tag("emu_from_mu")
            )),
            tags={"single_trigger", "single_e", "channel_e_tau"},
        ),

        #
        # single muon
        #
        Trigger(
            name="HLT_IsoMu24",
            id=105,
            legs=dict(
                mu=TriggerLeg(
                    pdg_id=13,
                    # min_pt=26.0,
                    # filter names:
                    # hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p08 (1mu + Iso)
                    # trigger_bits=Iso + SingleMuon,
                    trigger_bits=2 + 8,
                ),
            ),
            applies_to_dataset=(lambda dataset_inst: (
                dataset_inst.is_mc or
                dataset_inst.has_tag("mutau") or
                dataset_inst.has_tag("emu_from_e") or
                dataset_inst.has_tag("emu_from_mu") or
                dataset_inst.has_tag("mumu")
            )),
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        ),

        #
        # e tauh
        #
        Trigger(
            name="HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
            id=405,
            legs=dict(
                e=TriggerLeg(
                    pdg_id=11,
                    # min_pt=25.0,
                    # filter names:
                    # hltHpsOverlapFilterIsoEle24WPTightGsfLooseETauWPDeepTauPFTau30 (OverlapFilter, EleTau)
                    # trigger_bits= OverlapFilterPFTau + EleTau, # 8 +128 for v14
                    trigger_bits=8 + 64,
                ),
                tau=TriggerLeg(
                    pdg_id=15,
                    # min_pt=35.0,
                    # filter names:
                    # (DeepTau + HPS + OverlapFilterIsoEle)
                    # hltHpsOverlapFilterIsoEle24WPTightGsfLooseETauWPDeepTauPFTau30
                    # full trigger_bits= Tight + DeepTau + ETauFilter + OverlapFilterIsoEle + OverlapFilter + HPS # 4 + 8 + 128 + 4096 + 16777216 + 268435456 for v14  # noqa
                    # well Loose would work too v14 but it's the Electron that's loose so it's just dumb...
                    # actually needed:
                    # trigger_bits=DeepTau + HPS + OverlapFilterIsoEle # 8 + 32 + 256 for v12,
                    # trigger_bits=DeepTau + ETauFilter + OverlapFilterIsoEle # 8 + 128 + 4096 for v14,
                    # proposal: DeepTau + HPS + OverlapFilterIsoEle + if_exist(ETauFilter)
                    trigger_bits=8 + 32 + 256,
                ),
            ),
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.has_tag("etau")),
            tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        ),

        #
        # mu tauh
        #
        Trigger(
            name="HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1",
            id=304,
            legs=dict(
                mu=TriggerLeg(
                    pdg_id=13,
                    # min_pt=22.0,
                    # filter names:
                    # hltHpsOverlapFilterIsoMu20LooseMuTauWPDeepTauPFTau27L1Seeded (OverlapFilter PFTau, MuTau)
                    # trigger_bits= OverlapFilterPFTau + MuTau, # 4 + 64
                    trigger_bits=4 + 64,
                ),
                tau=TriggerLeg(
                    pdg_id=15,
                    # min_pt=32.0,
                    # filter names:
                    # (DeepTau + HPS + OverlapFilterIsoMu + L1Seeded)
                    # hltHpsOverlapFilterIsoMu20LooseMuTauWPDeepTauPFTau27L1Seeded
                    # full trigger_bits= DeepTau + MuTauFilter + OverlapFilterIsoMu + OverlapFilter + MatchL1HLT + HPS # 8 + 256 + 8192 + 16777216 + 134217728 + 268435456 for v14  # noqa
                    # well Loose would work too v14 but it's the Muon that's loose so it's just dumb...
                    # actually needed:
                    # trigger_bits=DeepTau + HPS + OverlapFilterIsoMu + L1Seeded # 8 + 32 + 512 + 262144 for v12,
                    # trigger_bits=DeepTau + MuTauFilter + OverlapFilterIsoMu + MatchL1HLT + 256 + 8192 + 134217728 for v14,  # noqa
                    # proposal: DeepTau + HPS + OverlapFilterIsoMu + if_exist(MuTauFilter) + L1Seeded and create L1Seeded for v14  # noqa
                    trigger_bits=8 + 32 + 512 + 262144,
                ),
            ),
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.has_tag("mutau")),
            tags={"cross_trigger", "cross_mu_tau", "channel_mu_tau"},
        ),

        #
        # tauh tauh
        #
        Trigger(
            name="HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1",
            id=507,
            legs=dict(
                tau1=TriggerLeg(
                    pdg_id=15,
                    # min_pt=40.0,
                    # filter names:
                    # hltHpsDoublePFTau35MediumDitauWPDeepTauL1HLTMatched (Deeptau + HPS + DeepTauDiTau)
                    # full trigger_bits= Medium + DeepTau + DiTau +  MatchL1HLT + HPS # 2 + 8 + 2048 + 134217728 + 268435456 for v14  # noqa
                    # proposal: DeepTauDiTau + HPS and create DeepTauDiTau for v14
                    trigger_bits=8 + 32 + 128,
                ),
                tau2=TriggerLeg(
                    pdg_id=15,
                    # min_pt=40.0,
                    # filter names:
                    # hltHpsDoublePFTau35MediumDitauWPDeepTauL1HLTMatched (Deeptau + HPS + DeepTauDiTau)
                    # same as above
                    trigger_bits=8 + 32 + 128,
                ),
            ),
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.has_tag("tautau")),
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),

        #
        # vbf
        #
        Trigger(
            name="HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1",
            id=602,
            legs=dict(
                tau1=TriggerLeg(
                    pdg_id=15,
                    # min_pt=25.0,
                    # filter names:
                    # (DeepTau + HPS + run 3 VBF+ditau)  # TODO: remove run 3 VBF+ditau, doesn't match
                    # hltHpsDoublePFTau20TrackDeepTauDitauWPForVBFIsoTau
                    # full trigger_bits= DeepTau + VBFDiTau + HPS # 8 + 1024 + 268435456 for v14
                    # proposal: if_exist(VBFDiTau) + HPS + DeepTau (last two actually useless for v14)
                    trigger_bits=8 + 32 + 4096,
                ),
                tau2=TriggerLeg(
                    pdg_id=15,
                    # min_pt=25.0,
                    # filter names:
                    # hltHpsDoublePFTau20TrackDeepTauDitauWPForVBFIsoTau
                    trigger_bits=8 + 32 + 4096,  # TODO: remove run 3 VBF+ditau, doesn't match
                ),
                # additional leg infos for vbf jets
                vbf1=TriggerLeg(
                    pdg_id=1,
                    # min_pt=115.0,
                    # filter names:
                    # The filters are applied to the lepton
                    # Taking the loosest filter for the Jets with the pt cut

                    # maybe hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20?
                    # (VBF cross-cleaned from medium deeptau PFTau)
                    # trigger_bits=VBFcrossCleanedDeepTauPFTau # 262144,  # does not work in v12 and v13  # TODO: add it for v14  # noqa
                ),
                vbf2=TriggerLeg(
                    pdg_id=1,
                    # min_pt=40.0,
                    # filter names:
                    # The filters are applied to the lepton

                    # maybe hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20?
                    # (VBF cross-cleaned from medium deeptau PFTau)
                    # trigger_bits=VBFcrossCleanedDeepTauPFTau # 262144,  # does not work in v12 and v13  # TODO: add it for v14  # noqa
                ),
            ),
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.has_tag("tautau")),
            tags={"cross_trigger", "cross_tau_tau_vbf", "channel_tau_tau"},
        ),

        # Currently disabled since it may not be needed
        # Trigger(
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
        Trigger(
            name="HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60",
            id=701,
            legs=dict(
                tau1=TriggerLeg(
                    pdg_id=15,
                    # min_pt=35.0,
                    # filter names:
                    # (DeepTau + Hps + ditau+PFJet)
                    # hltHpsOverlapFilterDeepTauDoublePFTau30PFJet60
                    # full trigger_bits= DeepTau + DiTauAndPFJet + OverlapFilter + HPS # 8 + 16384 + 16777216 + 268435456 for v14  # noqa
                    # proposal: DiTauAndPFJet # 16384

                    trigger_bits=8 + 32 + 16384,
                ),
                tau2=TriggerLeg(
                    pdg_id=15,
                    # min_pt=35.0,
                    # filter names:
                    # hltHpsOverlapFilterDeepTauDoublePFTau30PFJet60
                    trigger_bits=8 + 32 + 16384,
                ),
                jet=TriggerLeg(
                    pdg_id=1,
                    # min_pt=65.0,
                    # filter names:
                    # Filters are applied to the leptons
                    # Taking the loosest filter for the Jets with the pt cut

                    # hltHpsOverlapFilterDeepTauDoublePFTau30PFJet60
                    # (DoubleTau+Jet) -> 17
                    # trigger_bits=DoubleTau+Jet,
                    trigger_bits=131072,
                ),
            ),
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.has_tag("tautau")),
            tags={"cross_trigger", "cross_tau_tau_jet", "channel_tau_tau"},
        ),
    ])


def add_triggers_2023(config: od.Config) -> None:
    """
    Adds all triggers to a *config*. For the conversion from filter names to trigger bits, see
    https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py.
    """
    # get trigger bits for the requested nano version
    # bits = trigger_bits[config.campaign.x.version]

    config.x.triggers = od.UniqueObjectIndex(Trigger, [
        #
        # single electron
        #
        Trigger(
            name="HLT_Ele30_WPTight_Gsf",
            id=205,
            legs=dict(
                e=TriggerLeg(
                    pdg_id=11,
                    # min_pt=31.0,
                    # filter names:
                    # WPTightTrackIso
                    trigger_bits=2,
                ),
            ),
            applies_to_dataset=(lambda dataset_inst: (
                dataset_inst.is_mc or
                dataset_inst.has_tag("etau") or
                dataset_inst.has_tag("emu_from_e") or
                dataset_inst.has_tag("emu_from_mu")
            )),
            tags={"single_trigger", "single_e", "channel_e_tau"},
        ),

        #
        # single muon
        #
        Trigger(
            name="HLT_IsoMu24",
            id=105,
            legs=dict(
                mu=TriggerLeg(
                    pdg_id=13,
                    # min_pt=25.0,
                    # filter names:
                    # hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p08 (1mu + Iso)
                    trigger_bits=2 + 8,
                ),
            ),
            applies_to_dataset=(lambda dataset_inst: (
                dataset_inst.is_mc or
                dataset_inst.has_tag("mutau") or
                dataset_inst.has_tag("emu_from_e") or
                dataset_inst.has_tag("emu_from_mu") or
                dataset_inst.has_tag("mumu")
            )),
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        ),

        #
        # e tauh
        #
        Trigger(
            name="HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
            id=405,
            legs=dict(
                e=TriggerLeg(
                    pdg_id=11,
                    # min_pt=25.0,
                    # filter names:
                    # hltEle24erWPTightGsfTrackIsoFilterForTau
                    # hltHpsOverlapFilterIsoEle24WPTightGsfLooseETauWPDeepTauPFTau30 (OverlapFilter)
                    trigger_bits=8,
                ),
                tau=TriggerLeg(
                    pdg_id=15,
                    # min_pt=35.0,
                    # filter names:
                    # hltHpsSelectedPFTau30LooseETauWPDeepTauL1HLTMatched (DeepTau + HPS)
                    # hltHpsOverlapFilterIsoEle24WPTightGsfLooseETauWPDeepTauPFTau30
                    trigger_bits=8 + 32,
                ),
            ),
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.has_tag("etau")),
            tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        ),

        #
        # mu tauh
        #
        Trigger(
            name="HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1",
            id=304,
            legs=dict(
                mu=TriggerLeg(
                    pdg_id=13,
                    # min_pt=21.0,
                    # filter names:
                    # hltL3crIsoBigORMu18erTauXXer2p1L1f0L2f10QL3f20QL3trkIsoFiltered0p08
                    # hltHpsOverlapFilterIsoMu20LooseMuTauWPDeepTauPFTau27L1Seeded (OverlapFilter PFTau)
                    trigger_bits=4,
                ),
                tau=TriggerLeg(
                    pdg_id=15,
                    # min_pt=32.0,
                    # filter names:
                    # hltHpsSelectedPFTau27LooseMuTauWPDeepTauVsJetsAgainstMuonL1HLTMatched (DeepTau + HPS)
                    # hltHpsOverlapFilterIsoMu20LooseMuTauWPDeepTauPFTau27L1Seeded
                    trigger_bits=8 + 32,
                ),
            ),
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.has_tag("mutau")),
            tags={"cross_trigger", "cross_mu_tau", "channel_mu_tau"},
        ),

        #
        # tauh tauh
        #
        Trigger(
            name="HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1",
            id=507,
            legs=dict(
                tau1=TriggerLeg(
                    pdg_id=15,
                    # min_pt=40.0,
                    # filter names:
                    # hltHpsDoublePFTau35MediumDitauWPDeepTauL1HLTMatched (Deeptau + HPS)
                    trigger_bits=8 + 32,
                ),
                tau2=TriggerLeg(
                    pdg_id=15,
                    # min_pt=40.0,
                    # filter names:
                    # hltHpsDoublePFTau35MediumDitauWPDeepTauL1HLTMatched (Deeptau + HPS)
                    trigger_bits=8 + 32,
                ),
            ),
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.has_tag("tautau")),
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),

        #
        # vbf
        #
        Trigger(
            name="HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1",
            id=602,
            legs=dict(
                tau1=TriggerLeg(
                    pdg_id=15,
                    # min_pt=25.0,
                    # filter names:
                    # (DeepTau + HPS + run 3 VBF+ditau)
                    # hltHpsDoublePFTau20TrackDeepTauDitauWPForVBFIsoTau
                    # hltMatchedVBFOnePFJet2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20
                    # hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20
                    trigger_bits=8 + 32 + 4096,
                ),
                tau2=TriggerLeg(
                    pdg_id=15,
                    # min_pt=25.0,
                    # filter names:
                    # hltHpsDoublePFTau20TrackDeepTauDitauWPForVBFIsoTau
                    # hltMatchedVBFOnePFJet2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20
                    # hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20
                    trigger_bits=8 + 32 + 4096,
                ),
                # TODO: check if vbf legs are needed
            ),
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.has_tag("tautau")),
            tags={"cross_trigger", "cross_tau_tau_vbf", "channel_tau_tau"},
        ),

        #
        # tau tau jet
        #
        Trigger(
            name="HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60",
            id=701,
            legs=dict(
                tau1=TriggerLeg(
                    pdg_id=15,
                    # min_pt=35.0,
                    # filter names:
                    # (TightOOSCPhotons + di-tau + PFJet)
                    # hltHpsDoublePFTau30MediumDitauWPDeepTauL1HLTMatchedDoubleTauJet
                    trigger_bits=16 + 16384,
                ),
                tau2=TriggerLeg(
                    pdg_id=15,
                    # min_pt=35.0,
                    # filter names:
                    # hltHpsDoublePFTau30MediumDitauWPDeepTauL1HLTMatchedDoubleTauJet
                    trigger_bits=16 + 16384,
                ),
                # TODO: is this correct? copied from 2022
                jet=TriggerLeg(
                    pdg_id=1,
                    # min_pt=65.0,
                    # filter names:
                    # Filters are applied to the leptons
                    # Taking the loosest filter for the Jets with the pt cut

                    # hltHpsOverlapFilterDeepTauDoublePFTau30PFJet60
                    # (DoubleTau+Jet) -> 17
                    # trigger_bits=DoubleTau+Jet,
                    trigger_bits=131072,
                ),
            ),
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.has_tag("tautau")),
            tags={"cross_trigger", "cross_tau_tau_jet", "channel_tau_tau"},
        ),
    ])
