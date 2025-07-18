#!/usr/bin/env bash

# Script that sets up a virtual env in $CF_VENV_PATH.


action() {
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    # set variables and source the generic venv setup
    export CF_SANDBOX_FILE="${CF_SANDBOX_FILE:-${this_file}}"
    # export CF_SANDBOX_PATH="$( cd "$( dirname "${CF_SANDBOX_FILE}" )" && pwd )"
    export CF_VENV_NAME="$( basename "${this_file%.sh}" )"
    export CF_VENV_REQUIREMENTS="${CF_BASE}/sandboxes/columnar.txt,${CF_BASE}/sandboxes/dev.txt"

    source "${CF_BASE}/sandboxes/_setup_venv.sh" "$@" || return "$?"

    # setup kinfit
    source "${this_dir}/_setup_kinfit.sh" ""
}
action "$@"