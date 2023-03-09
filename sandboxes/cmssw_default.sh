#!/usr/bin/env bash

# Script that sets up a CMSSW environment in $CF_CMSSW_BASE.
# For more info on functionality and parameters, see the generic setup script _setup_cmssw.sh.

action() {
    # do the exact same setup as cf's version of this sandbox
    source "${CF_BASE}/sandboxes/cmssw_default.sh" "$@"

    # change the default wlcg fs in this sandbox
    export LAW__target__default_wlcg_fs="wlcg_fs_desy_gsiftp"
}
action "$@"
