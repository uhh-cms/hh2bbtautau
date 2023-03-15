#!/usr/bin/env bash

# Script that sets up a CMSSW environment in $CF_CMSSW_BASE.
# For more info on functionality and parameters, see the generic setup script _setup_cmssw.sh.

action() {
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"

    export CF_SANDBOX_FILE="${CF_SANDBOX_FILE:-${this_file}}"
    export LAW__target__default_wlcg_fs="wlcg_fs_desy_gsiftp"

    # do the exact same setup as cf's version of this sandbox
    source "${CF_BASE}/sandboxes/cmssw_default.sh" "$@"
}
action "$@"
