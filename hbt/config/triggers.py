# coding: utf-8

"""
Definition of triggers
"""

import order as od

from hbt.config.util import Trigger, TriggerLeg

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
            id=710,  # TODO
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=26.0,  # TODO
                    # filter names:
                    #
                    trigger_bits=None,  # TODO
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=22.0,  # TODO
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
            id=711,  # TODO
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=26.0,  # TODO
                    # filter names:
                    #
                    trigger_bits=None,  # TODO
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=22.0,  # TODO
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
            id=712,  # TODO
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=26.0,  # TODO
                    # filter names:
                    #
                    trigger_bits=None,  # TODO
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=32.0,  # TODO
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
            id=706,  # TODO
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    min_pt=22,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=23,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
            ],
            tags={"cross_trigger", "cross_mu_tau", "channel_mu_tau"},
        ),
        Trigger(
            name="HLT_IsoMu19_eta2p1_LooseIsoPFTau20_SingleL1",
            id=707,  # TODO
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    min_pt=22,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=23,  # TODO
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
            id=708,  # TODO
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    min_pt=38,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=38,  # TODO
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
            id=709,  # TODO
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    min_pt=38,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=38,  # TODO
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
            id=701,  # TODO
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=28,  # TODO
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
            id=702,  # TODO
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    min_pt=25,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
            ],
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        )
        config.x.triggers.add(
            name="HLT_IsoMu22_eta2p1",
            id=703,  # TODO
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    min_pt=25,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
            ],
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        )
        config.x.triggers.add(
            name="HLT_IsoTkMu22",
            id=704,  # TODO
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    min_pt=25,  # TODO
                    # filter names:
                    # TODO
                    trigger_bits=None,  # TODO
                ),
            ],
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        )
        config.x.triggers.add(
            name="HLT_IsoTkMu22_eta2p1",
            id=705,  # TODO
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    min_pt=25,  # TODO
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
            id=201,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=35.0,
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
            id=202,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=35.0,
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
            id=203,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=38.0,
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
            id=101,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    min_pt=26.0,
                    # filter names:
                    # hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p07
                    trigger_bits=2,
                ),
            ],
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        ),
        Trigger(
            name="HLT_IsoMu27",
            id=102,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    min_pt=29.0,
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
            id=401,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=27.0,
                    # filter names:
                    # hltEle24erWPTightGsfTrackIsoFilterForTau
                    # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
                    trigger_bits=2 + 64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=35.0,
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
            id=301,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    min_pt=22.0,
                    # filter names:
                    # hltL3crIsoL1sMu18erTau24erIorMu20erTau24erL1f0L2f10QL3f20QL3trkIsoFiltered0p07
                    # hltOverlapFilterIsoMu20LooseChargedIsoPFTau27L1Seeded
                    trigger_bits=2 + 64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=32.0,
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
            id=501,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    min_pt=40.0,
                    # filter names:
                    # hltDoublePFTau35TrackPt1MediumChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=40.0,
                    # filter names:
                    # hltDoublePFTau35TrackPt1MediumChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
            ],
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),
        Trigger(
            name="HLT_DoubleTightChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg",
            id=502,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    min_pt=40.0,
                    # filter names:
                    # hltDoublePFTau35TrackPt1TightChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=40.0,
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
            id=503,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    min_pt=45.0,
                    # filter names:
                    # hltDoublePFTau40TrackPt1MediumChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=45.0,
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
            id=504,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    min_pt=45.0,
                    # filter names:
                    # hltDoublePFTau40TrackPt1TightChargedIsolationDz02Reg
                    trigger_bits=64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=45.0,
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
                    min_pt=25.0,
                    # filter names:
                    # hltDoublePFTau20TrackPt1LooseChargedIsolation
                    trigger_bits=2048,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=25.0,
                    # filter names:
                    # hltDoublePFTau20TrackPt1LooseChargedIsolation
                    trigger_bits=2048,
                ),
                # additional leg infos for vbf jets
                # TODO check if vbf legs are needed
                TriggerLeg(
                    min_pt=115.0,
                    # filter names:
                    # hltMatchedVBFOnePFJet2CrossCleanedFromDoubleLooseChargedIsoPFTau20
                    trigger_bits=1,
                ),
                TriggerLeg(
                    min_pt=40.0,
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
            id=201,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=35.0,
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
            id=203,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=38.0,
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
            id=101,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    min_pt=26.0,
                    # filter names:
                    # hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p07
                    trigger_bits=2,
                ),
            ],
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        ),
        Trigger(
            name="HLT_IsoMu27",
            id=102,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    min_pt=29.0,
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
            id=401,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=27.0,
                    # filter names:
                    # hltEle24erWPTightGsfTrackIsoFilterForTau
                    # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
                    trigger_bits=2 + 64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=35.0,
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
            id=301,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    min_pt=22.0,
                    # filter names:
                    # hltL3crIsoL1sMu18erTau24erIorMu20erTau24erL1f0L2f10QL3f20QL3trkIsoFiltered0p07
                    # hltOverlapFilterIsoMu20LooseChargedIsoPFTau27L1Seeded
                    trigger_bits=2 + 64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=32.0,
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
            id=502,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    min_pt=40.0,
                    # filter names:
                    # hltDoublePFTau35TrackPt1TightChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=40.0,
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
            id=503,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    min_pt=45.0,
                    # filter names:
                    # hltDoublePFTau40TrackPt1MediumChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=45.0,
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
            id=504,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    min_pt=45.0,
                    # filter names:
                    # hltDoublePFTau40TrackPt1TightChargedIsolationDz02Reg
                    trigger_bits=64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=45.0,
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
                    min_pt=25.0,
                    # filter names:
                    # hltDoublePFTau20TrackPt1LooseChargedIsolation
                    trigger_bits=2048,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=25.0,
                    # filter names:
                    # hltDoublePFTau20TrackPt1LooseChargedIsolation
                    trigger_bits=2048,
                ),
                # additional leg infos for vbf jets
                TriggerLeg(
                    min_pt=115.0,
                    # filter names:
                    # hltMatchedVBFOnePFJet2CrossCleanedFromDoubleLooseChargedIsoPFTau20
                    trigger_bits=1,
                ),
                TriggerLeg(
                    min_pt=40.0,
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
            id=201,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=31.0,
                    # filter names:
                    # hltEle30WPTightGsfTrackIsoFilter
                    trigger_bits=2,
                ),
            ],
            tags={"single_trigger", "single_e", "channel_e_tau"},
        ),
        # Currently disabled
        # Trigger(
        #     name="HLT_Ele35_WPTight_Gsf",
        #     id=203,
        #     legs=[
        #         TriggerLeg(
        #             pdg_id=11,
        #             min_pt=36.0,
        #             # filter names:
        #             # hltEle35noerWPTightGsfTrackIsoFilter (Single e WPTight)
        #             trigger_bits=2,
        #         ),
        #     ],
        #     tags={"single_trigger", "single_e", "channel_e_tau"},
        # ),

        #
        # single muon
        #
        Trigger(
            name="HLT_IsoMu24",
            id=101,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    min_pt=26.0,
                    # filter names:
                    # hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p08 (1mu + Iso)
                    trigger_bits=2 + 8,
                ),
            ],
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        ),
        # Currently disabled
        # Trigger(
        #     name="HLT_IsoMu27",
        #     id=102,
        #     legs=[
        #         TriggerLeg(
        #             pdg_id=13,
        #             min_pt=28.0,
        #             # filter names:
        #             # hltL3crIsoL1sMu22Or25L1f0L2f10QL3f27QL3trkIsoFiltered0p08 (1mu + Iso)
        #             trigger_bits=2 + 8,
        #         ),
        #     ],
        #     tags={"single_trigger", "single_mu", "channel_mu_tau"},
        # ),

        #
        # e tauh
        #
        Trigger(
            name="HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
            id=401,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=25.0,
                    # filter names:
                    # hltOverlapFilterIsoEle24IsoTau30WPTightGsfCaloJet5 # TODO Twiki has no matches
                    # hltHpsOverlapFilterIsoEle24WPTightGsfLooseETauWPDeepTauPFTau30 (OverlapFilter) # TODO Twiki sugests 8 + 64  # noqa
                    trigger_bits=8,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=35.0,
                    # filter names:
                    # (DeepTau + HPS) # TODO Twiki sugests 8 + 32 + 256
                    # hltHpsOverlapFilterIsoEle24WPTightGsfLooseETauWPDeepTauPFTau30
                    trigger_bits=8 + 32,
                ),
            ],
            tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        ),

        #
        # mu tauh
        #
        Trigger(
            name="HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1",
            id=301,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    min_pt=22.0,
                    # filter names:
                    # hltL3crIsoBigORMu18erTauXXer2p1L1f0L2f10QL3f20QL3trkIsoFiltered  # TODO Twiki sugests 2
                    # hltHpsOverlapFilterIsoMu20LooseMuTauWPDeepTauPFTau27L1Seeded (OverlapFilter PFTau) # TODO Twiki sugests 4 + 64  # noqa
                    trigger_bits=4,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=32.0,
                    # filter names:
                    # (DeepTau + HPS) # TODO Twiki sugests 8 + 32 + 512 + 262144
                    # hltHpsOverlapFilterIsoMu20LooseMuTauWPDeepTauPFTau27L1Seeded
                    trigger_bits=8 + 32,
                ),
            ],
            tags={"cross_trigger", "cross_mu_tau", "channel_mu_tau"},
        ),

        #
        # tauh tauh
        #
        Trigger(
            name="HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1",
            id=505,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    min_pt=40.0,
                    # filter names:
                    # hltHpsDoublePFTau35MediumDitauWPDeepTauDz02 (Deeptau + HPS)
                    trigger_bits=8 + 32,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=40.0,
                    # filter names:
                    # hltHpsDoublePFTau35MediumDitauWPDeepTauDz02 (Deeptau + HPS)
                    trigger_bits=8 + 32,
                ),
            ],
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),
        # Currently disabled
        # Trigger(
        #     name="HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1",
        #     id=506,
        #     legs=[
        #         TriggerLeg(
        #             pdg_id=15,
        #             min_pt=45.0,
        #             # filter names:
        #             trigger_bits=2 + 32 + 64,
        #         ),
        #         TriggerLeg(
        #             pdg_id=15,
        #             min_pt=45.0,
        #             # filter names:
        #             trigger_bits=2 + 32 + 64,
        #         ),
        #     ],
        #     applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "E"),
        #     tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        # ),

        # Trigger(
        #     name="HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1",
        #     id=507,
        #     legs=[
        #         TriggerLeg(
        #             pdg_id=15,
        #             min_pt=45.0,
        #             # filter names:
        #             # hltHpsDoubleMediumChargedIsoDisplPFTau32Dxy0p005
        #             trigger_bits=2 + 32 + 64 + 32768,
        #         ),
        #         TriggerLeg(
        #             pdg_id=15,
        #             min_pt=45.0,
        #             # filter names:
        #             # hltHpsDoubleMediumChargedIsoDisplPFTau32Dxy0p005
        #             trigger_bits=2 + 32 + 64 + 32768,
        #         ),
        #     ],
        #     applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data and dataset_inst.x.era < "E"),
        #     tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        # ),

        #
        # vbf
        #
        Trigger(
            name="HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1",
            id=603,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    min_pt=25.0,
                    # filter names:
                    # (DeepTau + HPS + run 3 VBF+ditau) # TODO Twiki sugests 8
                    # hltMatchedVBFOnePFJet2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20
                    trigger_bits=8 + 32 + 4096,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=25.0,
                    # filter names:
                    # hltMatchedVBFOnePFJet2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20
                    trigger_bits=8 + 32 + 4096,
                ),
                # # additional leg infos for vbf jets
                # TriggerLeg(
                #     min_pt=115.0,
                #     # filter names:
                #     # The filters are applied to the lepton
                #     # Taking the loosest filter for the Jets with the pt cut
                #     trigger_bits=1,
                # ),
                # TriggerLeg(
                #     min_pt=40.0,
                #     # filter names:
                #     # The filters are applied to the lepton
                #     trigger_bits=1,
                # ),
            ],
            # applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data and config.has_tag("pre")),
            tags={"cross_trigger", "cross_tau_tau_vbf", "channel_tau_tau"},
        ),

        # currently disabled
        # Trigger(
        #     name="HLT_VBF_DoubleLooseChargedIsoPFTauHPS20_Trk1_eta2p1",
        #     id=602,
        #     legs=[
        #         TriggerLeg(
        #             pdg_id=15,
        #             min_pt=25.0,
        #             # filter names:
        #             # hltMatchedVBFOnePFJet2CrossCleanedFromDoubleLooseChargedIsoPFTauHPS20 (LooseChargedIso + HPS + run 3 VBF+ditau) # noqa
        #             # hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleLooseChargedIsoPFTauHPS20
        #             # hltHpsDoublePFTau20TrackLooseChargedIso
        #             trigger_bits=1 + 32 + 4096,
        #         ),
        #         TriggerLeg(
        #             pdg_id=15,
        #             min_pt=25.0,
        #             # filter names:
        #             # hltMatchedVBFOnePFJet2CrossCleanedFromDoubleLooseChargedIsoPFTauHPS20 (LooseChargedIso + HPS + run 3 VBF+ditau)  # noqa
        #             # hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleLooseChargedIsoPFTauHPS20
        #             # hltHpsDoublePFTau20TrackLooseChargedIso
        #             trigger_bits=1 + 32 + 4096,
        #         ),
        #         # additional leg infos for vbf jets
        #         TriggerLeg(  # TODO
        #             min_pt=115.0,
        #             # filter names:
        #             trigger_bits=None,
        #         ),
        #         TriggerLeg(
        #             min_pt=40.0,
        #             # filter names:
        #             trigger_bits=None,
        #         ),
        #     ],
        #     applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc and config.has_tag("pre")),
        #     tags={"cross_trigger", "cross_tau_tau_vbf", "channel_tau_tau"},
        # ),

        # Currently disabled since it may not be needed
        # Trigger(
        #     name="HLT_DoublePFJets40_Mass500_MediumDeepTauPFTauHPS45_L2NN_MediumDeepTauPFTauHPS20_eta2p1",
        #     id=604,
        #     legs=[
        #         TriggerLeg(
        #             pdg_id=15,
        #             min_pt=25.0,
        #             trigger_bits=None,
        #         ),
        #         TriggerLeg(
        #             pdg_id=15,
        #             min_pt=25.0,
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
                    min_pt=35.0,
                    # filter names:
                    # (TightOOSCPhotons + di-tau + PFJet) # TODO Twiki sugests 8 + 32 + 16384
                    # hltHpsOverlapFilterDeepTauDoublePFTau30PFJet60
                    trigger_bits=16 + 16384,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=35.0,
                    # filter names:
                    # hltHpsOverlapFilterDeepTauDoublePFTau30PFJet60
                    trigger_bits=16 + 16384,
                ),
                # TriggerLeg(
                #     min_pt=65.0,
                #     # filter names:
                #     # Filters are applied to the leptons
                #     # Taking the loosest filter for the Jets with the pt cut
                #     trigger_bits=1,
                # ),
            ],
            tags={"cross_trigger", "cross_tau_tau_jet", "channel_tau_tau"},
        ),
        # Currently disabled
        # Trigger(
        #     name="HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet75",
        #     id=702,
        #     legs=[
        #         TriggerLeg(
        #             pdg_id=15,
        #             min_pt=35.0,
        #             # filter names:
        #             # hltHpsOverlapFilterDeepTauDoublePFTau30PFJet75 (Deeptau + HPS + di-tau + PFJet)
        #             # hltHpsDoublePFTau30MediumDitauWPDeepTauL1HLTMatchedDoubleTauJet
        #             # hlt1PFJet75L1HLTMatched
        #             trigger_bits=8 + 32 + 16384,
        #         ),
        #         TriggerLeg(
        #             pdg_id=15,
        #             min_pt=35.0,
        #             # filter names:
        #             # hltHpsOverlapFilterDeepTauDoublePFTau30PFJet75
        #             # hltHpsDoublePFTau30MediumDitauWPDeepTauL1HLTMatchedDoubleTauJet
        #             # hlt1PFJet75L1HLTMatched
        #             trigger_bits=8 + 32 + 16384,
        #         ),
        #         TriggerLeg(
        #             min_pt=75.0,
        #             # filter names:
        #             # hltHpsOverlapFilterDeepTauDoublePFTau30PFJet75
        #             trigger_bits=None,
        #         ),
        #     ],
        #     tags={"cross_trigger", "cross_tau_tau_jet", "channel_tau_tau"},
        # ),
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
            id=201,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=31.0,
                    # filter names:
                    # WPTightTrackIso
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
            id=101,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    min_pt=25.0,
                    # filter names:
                    # hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p08 (1mu + Iso)
                    trigger_bits=2 + 8,
                ),
            ],
            tags={"single_trigger", "single_mu", "channel_mu_tau"},
        ),

        #
        # e tauh
        #
        Trigger(
            name="HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
            id=401,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=25.0,
                    # filter names:
                    # hltEle24erWPTightGsfTrackIsoFilterForTau
                    # hltHpsOverlapFilterIsoEle24WPTightGsfLooseETauWPDeepTauPFTau30 (OverlapFilter)
                    trigger_bits=8,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=35.0,
                    # filter names:
                    # hltHpsSelectedPFTau30LooseETauWPDeepTauL1HLTMatched (DeepTau + HPS)
                    # hltHpsOverlapFilterIsoEle24WPTightGsfLooseETauWPDeepTauPFTau30
                    trigger_bits=8 + 32,
                ),
            ],
            tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        ),

        #
        # mu tauh
        #
        Trigger(
            name="HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1",
            id=301,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    min_pt=21.0,
                    # filter names:
                    # hltL3crIsoBigORMu18erTauXXer2p1L1f0L2f10QL3f20QL3trkIsoFiltered0p08
                    # hltHpsOverlapFilterIsoMu20LooseMuTauWPDeepTauPFTau27L1Seeded (OverlapFilter PFTau)
                    trigger_bits=4,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=32.0,
                    # filter names:
                    # hltHpsSelectedPFTau27LooseMuTauWPDeepTauVsJetsAgainstMuonL1HLTMatched (DeepTau + HPS)
                    # hltHpsOverlapFilterIsoMu20LooseMuTauWPDeepTauPFTau27L1Seeded
                    trigger_bits=8 + 32,
                ),
            ],
            tags={"cross_trigger", "cross_mu_tau", "channel_mu_tau"},
        ),

        #
        # tauh tauh
        #
        Trigger(
            name="HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1",
            id=505,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    min_pt=40.0,
                    # filter names:
                    # hltHpsDoublePFTau35MediumDitauWPDeepTauL1HLTMatched (Deeptau + HPS)
                    trigger_bits=8 + 32,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=40.0,
                    # filter names:
                    # hltHpsDoublePFTau35MediumDitauWPDeepTauL1HLTMatched (Deeptau + HPS)
                    trigger_bits=8 + 32,
                ),
            ],
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),

        #
        # vbf
        #
        Trigger(
            name="HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1",
            id=603,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    min_pt=25.0,
                    # filter names:
                    # (DeepTau + HPS + run 3 VBF+ditau)
                    # hltHpsDoublePFTau20TrackDeepTauDitauWPForVBFIsoTau
                    # hltMatchedVBFOnePFJet2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20
                    # hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20
                    trigger_bits=8 + 32 + 4096,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=25.0,
                    # filter names:
                    # hltHpsDoublePFTau20TrackDeepTauDitauWPForVBFIsoTau
                    # hltMatchedVBFOnePFJet2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20
                    # hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20
                    trigger_bits=8 + 32 + 4096,
                ),
            ],
            # applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data and config.has_tag("pre")),
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
                    min_pt=35.0,
                    # filter names:
                    # (TightOOSCPhotons + di-tau + PFJet)
                    # hltHpsDoublePFTau30MediumDitauWPDeepTauL1HLTMatchedDoubleTauJet
                    trigger_bits=16 + 16384,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=35.0,
                    # filter names:
                    # hltHpsDoublePFTau30MediumDitauWPDeepTauL1HLTMatchedDoubleTauJet
                    trigger_bits=16 + 16384,
                ),
            ],
            tags={"cross_trigger", "cross_tau_tau_jet", "channel_tau_tau"},
        ),
    ])
