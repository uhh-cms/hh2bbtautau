# HH → bb𝜏𝜏

[![Lint and test](https://github.com/uhh-cms/hh2bbtautau/actions/workflows/lint_and_test.yaml/badge.svg)](https://github.com/uhh-cms/hh2bbtautau/actions/workflows/lint_and_test.yaml)
[![License](https://img.shields.io/github/license/uhh-cms/hh2bbtautau.svg)](https://github.com/uhh-cms/hh2bbtautau/blob/master/LICENSE)


### Quickstart

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
    --dataset st_tchannel_t \
    --branch 0

# create a plot
law run cf.PlotVariables \
    --version v1 \
    --datasets st_tchannel_t \
    --producers features \
    --variables jet1_pt \
    --categories 1e \
    --branch 0

# create a (test) datacard (CMS-style)
law run cf.CreateDatacards \
    --version v1 \
    --producers features \
    --inference-model test \
    --workers 3
```


### Development

- Source hosted at [GitHub](https://github.com/uhh-cms/hh2bbtautau)
- Report issues, questions, feature requests on [GitHub Issues](https://github.com/uhh-cms/hh2bbtautau/issues)
