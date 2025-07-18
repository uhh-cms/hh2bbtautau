#!/usr/bin/env bash

action() {
    # setup kinfit here, downloading repo, installing into venv directories for include, lib, lib/python...
    #!/usr/bin/env bash


    local HHKINFIT2_DIR="${CF_VENV_BASE}/HHKinFit2"

    if [ ! -d "${HHKINFIT2_DIR}" ]; then
        echo "[sandbox] Cloning HHKinFit2..."
        git clone https://github.com/HEP-KBFI/HHKinFit2.git "${HHKINFIT2_DIR}" || return "$?"
    fi

    local SO_FILE_GLOB="${HHKINFIT2_DIR}/python/hhkinfit2.cpython-*.so"
    if ls ${SO_FILE_GLOB} 1>/dev/null 2>&1; then
        echo "Importing HHKinFit2."
    else
        echo "[sandbox] Building HHKinFit2..."
        local orig_dir="$(pwd)"
        cd "${HHKINFIT2_DIR}"
        chmod +x installpython.sh
        ./installpython.sh || return "$?"
        cd "${orig_dir}"
        echo "[sandbox] HHKinFit2 build complete."
    fi

    export PYTHONPATH="${HHKINFIT2_DIR}/python:${PYTHONPATH}"
}
action "$@"