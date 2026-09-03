#!/usr/bin/env bash

# Script demonstrating the pipeline "datacard -> workspsace -> fit diagnostics -> postfit shapes".

action() {
    # settings
    local name="hbt_bl"
    local unblinded=false
    local samples="100"
    local overwrite_ws=false
    local overwrite_fd=false

    # parse arguments
    local dc="$1"
    if [ ! -f "${dc}" ]; then
        echo "datacard ${dc} does not exist"
        return "1"
    fi

    # derived values
    local ws="$( basename "${dc%.txt}" ).root"
    ws="workspace__${ws#datacard__}"
    local fd="fitDiagnostics.${name}.root"
    local sf="out.${name}.root"

    ${overwrite_ws} && rm -f "${ws}"
    if [ ! -f "${ws}" ]; then
        echo -e "\x1b[0;49;32mcreating workspace ...\x1b[0m"
        text2workspace.py \
            "${dc}" \
            --out "${ws}" \
            --mass 125.0 \
            --optimize-simpdf-constraints cms \
            --physics-model dhi.models.hh_model_bbtt:model_default_run3 \
            --physics-option doklDependentUnc=True \
            --physics-option doBRscaling=True \
            --physics-option doHscaling=True \
        || return "$?"
        echo -e "\x1b[0;49;32mdone\x1b[0m"
    else
        echo -e "\x1b[0;49;32musing existing '${ws}'\x1b[0m"
    fi

    ${overwrite_fd} && rm -f "${fd}"
    if [ ! -f "${fd}" ]; then
        echo -e "\n\x1b[0;49;32mrunning fit diganostics\x1b[0m"
        combine \
            --method FitDiagnostics \
            "${ws}" \
            --verbose 1 \
            --mass 125.0 \
            --redefineSignalPOIs r \
            --setParameters r=1.0,r_gghh=1.0,r_qqhh=1.0,kl=1.0,kt=1.0,CV=1.0,C2V=1.0 \
            --freezeParameters r_gghh,r_qqhh,kl,kt,CV,C2V \
            --skipBOnlyFit \
            --cminDefaultMinimizerType Minuit2 \
            --cminDefaultMinimizerStrategy 0 \
            --cminDefaultMinimizerTolerance 0.1 \
            --cminFallbackAlgo Minuit2,0:0.2 \
            --cminFallbackAlgo Minuit2,0:0.4 \
            $( ${unblinded} && echo "--toys -1" ) \
            --name ".${name}" \
        || return "$?"
        echo -e "\x1b[0;49;32mdone\x1b[0m"
    else
        echo -e "\x1b[0;49;32musing existing '${fd}'\x1b[0m"
    fi

    echo -e "\n\x1b[0;49;32mcreating shapes with ${samples} samples ...\x1b[0m"
    PostFitShapesFromWorkspace \
        --datacard "${dc}"\
        --workspace "${ws}" \
        --fitresult "${fd}:fit_s" \
        --output "${sf}" \
        --postfit \
        --samples "${samples}" \
        --print \
    || return "$?"
    echo -e "\x1b[0;49;32mdone\x1b[0m"
}

action "$@"
