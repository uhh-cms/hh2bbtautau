# HH → bb𝜏𝜏

[![Lint and test](https://github.com/uhh-cms/hh2bbtautau/actions/workflows/lint_and_test.yaml/badge.svg)](https://github.com/uhh-cms/hh2bbtautau/actions/workflows/lint_and_test.yaml)

## Quickstart

A couple test tasks are listed below.
They might require a **valid voms proxy** for accessing input data.

```shell
# clone the project
git clone --recursive git@github.com:uhh-cms/hh2bbtautau.git
cd hh2bbtautau

# source the setup and store decisions in .setups/dev.sh (arbitrary name)
source setup.sh dev

# index existing tasks once to enable auto-completion for "law run"
law index --verbose

# run your first task
# (they are all shipped with columnflow and thus have the "cf." prefix)
law run cf.ReduceEvents \
    --version v1 \
    --dataset hh_ggf_bbtautau_madgraph \
    --branch 0

# create a plot
law run cf.PlotVariables1D \
    --version v1 \
    --datasets hh_ggf_bbtautau_madgraph \
    --producers default \
    --variables jet1_pt \
    --categories incl \
    --branch 0

# create a (test) datacard (CMS-style)
law run cf.CreateDatacards \
    --version v1 \
    --producers default \
    --inference-model test \
    --workers 3
```

## Useful commands

### Full reduction

```shell
law run cf.ReduceEventsWrapper \
    --version prod1 \
    --configs run3_2022_preEE \
    --datasets "*" \
    --shifts "nominal,{tune,hdamp,mtop}_{up,down}" \
    --cf.ReduceEvents-workflow htcondor \
    --cf.ReduceEvents-pilot \
    --cf.ReduceEvents-tasks-per-job 3 \
    --local-scheduler False \
    --workers 6
```

## Useful links

- [columnflow documentation](https://columnflow.readthedocs.io/en/latest/index.html)
- CMS services
  - [HLT info browser](https://cmshltinfo.app.cern.ch/path/HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1_v)
  - [HLT config browser](https://cmshltcfg.app.cern.ch/open?db=online&cfg=%2Fcdaq%2Fphysics%2FRun2018%2F2e34%2Fv2.1.5%2FHLT%2FV2)
  - [GrASP](https://cms-pdmv-prod.web.cern.ch/grasp/)
  - [XSDB](https://xsecdb-xsdb-official.app.cern.ch/xsdb)
  - [DAS](https://cmsweb.cern.ch/das)
NanoAOD:
  - [Nano documentation](https://gitlab.cern.ch/cms-nanoAOD/nanoaod-doc)
  - [Correctionlib files](https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration)
- JME
  - [Docs](https://cms-jerc.web.cern.ch)
- BTV
  - [Docs](https://btv-wiki.docs.cern.ch)
- TAU
  - [Run 2 Twiki](https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2)
  - [Run 3 Twiki](https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun3)
  - [Correctionlib files](https://gitlab.cern.ch/cms-tau-pog/jsonpog-integration/-/tree/TauPOG_v2_deepTauV2p5/POG/TAU?ref_type=heads)

## Development

- Source hosted at [GitHub](https://github.com/uhh-cms/hh2bbtautau)
- Report issues, questions, feature requests on [GitHub Issues](https://github.com/uhh-cms/hh2bbtautau/issues)
