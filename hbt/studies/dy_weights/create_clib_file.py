# coding: utf-8

"""
Toy script to explore the creation of a correctionlib file with multiple, mixed dependencies (categories, bin ranges,
and formulas) for the DY weight correction in the hh2bbtautau analysis.
"""

from __future__ import annotations
from typing import Literal

import re
import dataclasses
import law

import correctionlib.schemav2 as cs


# helpers

def expr_in_range(expr: str, lower_bound: float | int, upper_bound: float | int) -> str:
    # lower bound must be smaller than upper bound
    assert lower_bound < upper_bound
    return expr


def create_dy_weight_correction(dy_weight_data: dict) -> cs.Correction:
    # create the correction object
    dy_weight_correction = cs.Correction(
        name="dy_weight",
        description="DY weights derived in the phase space of the hh2bbtautau analysis, supposed to correct njet and "
        "ptll distributions, as well as correlated quantities.",
        version=1,
        inputs=[
            cs.Variable(name="era", type="string", description="Era name."),
            cs.Variable(name="njets", type="int", description="Number of jets in the event. To be capped at 10."),
            cs.Variable(name="ntags", type="int", description="Number of (PNet) b-tagged jets. To be capped at 10."),
            cs.Variable(name="ptll", type="real", description="pT of the reconstructed dilepton system in GeV."),
            cs.Variable(name="syst", type="string", description="Systematic variation."),
        ],
        output=cs.Variable(name="weight", type="real", description="DY event weight."),
        data=cs.Category(
            nodetype="category",
            input="era",
            content=[],
        ),
    )
    # dynamically fill it
    for era, era_data in dy_weight_data.items():
        era_category_content = []
        for syst, syst_data in era_data.items():
            njet_bin_content = []
            for (min_njet, max_njet), ntag_data in syst_data.items():
                ntag_bin_content = []
                for (min_ntag, max_ntag), formulas in ntag_data.items():
                    # create a joined expression for all formulas
                    expr = "+".join(
                        expr_in_range(formula, lower_bound, upper_bound)
                        for lower_bound, upper_bound, formula in formulas
                    )
                    # add formula object to binning content
                    ntag_bin_content.append(
                        cs.Formula(
                            nodetype="formula",
                            variables=["ptll"],
                            parser="TFormula",
                            expression=expr,
                        ),
                    )
                njet_bin_content.append(
                    cs.Binning(
                        nodetype="binning",
                        input="ntags",
                        flow="error",
                        edges=sorted(set(sum(map(list, ntag_data.keys()), []))),
                        content=ntag_bin_content,
                    ),
                )
            # add a new category item for the jet bins
            era_category_content.append(
                cs.CategoryItem(
                    key=syst,
                    value=cs.Binning(
                        nodetype="binning",
                        input="njets",
                        flow="error",
                        edges=sorted(set(sum(map(list, syst_data.keys()), []))),
                        content=njet_bin_content,
                    ),
                ),
            )
        # add a new category item for the era
        dy_weight_correction.data.content.append(
            cs.CategoryItem(
                key=era,
                value=cs.Category(
                    nodetype="category",
                    input="syst",
                    content=era_category_content,
                ),
            ),
        )
    return dy_weight_correction


if __name__ == "__main__":
    import gzip

    @dataclasses.dataclass
    class Norm:
        nom: float
        unc: float

        @property
        def up(self) -> float:
            return self.nom + self.unc

        @property
        def down(self) -> float:
            return max(0.0, self.nom - self.unc)

    # dilep_pt fit functions
    # era --> number of central jets --> post-fit formula string taking gen_pT as x variable
    dilep_pt_formulas = {
        "2022preEE": {
            2: "(0.5*(erf(-0.08*(x-29.35284418720058))+1))*((0.6000000000000001)+((4.13788351736806)*(1/12.68384416493451)*exp(-0.5*((x-16.549864140670717)/12.68384416493451)^2)))+(0.5*(erf(0.08*(x-29.35284418720058))+1))*((0.886372488501618)+(0.0002545066273632575)*x)",  # noqa: E501
            3: "(0.5*(erf(-0.08*(x-20.000000000006253))+1))*((0.6000000000055947)+((4.684935079638329)*(1/9.749315961972627)*exp(-0.5*((x-14.635668433059768)/9.749315961972627)^2)))+(0.5*(erf(0.08*(x-20.000000000006253))+1))*((0.9937281234059175)+(1.192308108942791e-05)*x)",  # noqa: E501
            4: "(0.5*(erf(-0.08*(x-24.02708186962039))+1))*((0.7901754443527945)+((5.158864554240971)*(1/8.358857902225806)*exp(-0.5*((x-15.914275456756133)/8.358857902225806)^2)))+(0.5*(erf(0.08*(x-24.02708186962039))+1))*((1.2694392595110884)+(-0.00043892270855478473)*x)",  # noqa: E501
        },
        "2022postEE": {
            2: "(0.5*(erf(-0.08*(x-32.145006781489))+1))*((0.6000000000000001)+((3.304254342680554)*(1/11.163631796783257)*exp(-0.5*((x-17.145687100915694)/11.163631796783257)^2)))+(0.5*(erf(0.08*(x-32.145006781489))+1))*((0.8850031265877162)+(0.00033950838587539986)*x)",  # noqa: E501
            3: "(0.5*(erf(-0.08*(x-22.054815668367077))+1))*((0.6000000000000001)+((4.1914827055520565)*(1/11.714555808339519)*exp(-0.5*((x-15.294417995696852)/11.714555808339519)^2)))+(0.5*(erf(0.08*(x-22.054815668367077))+1))*((0.9446099605080525)+(0.0003426496829583549)*x)",  # noqa: E501
            4: "(0.5*(erf(-0.08*(x-30.583320181625048))+1))*((0.6000000000000001)+((8.818375382937381)*(1/15.782567033596633)*exp(-0.5*((x-16.142725826324664)/15.782567033596633)^2)))+(0.5*(erf(0.08*(x-30.583320181625048))+1))*((1.1834791971110694)+(0.00012342913069317776)*x)",  # noqa: E501
        },
        "2023preBPix": {
            2: "(0.5*(erf(-0.08*(x-20.000000000000004))+1))*((0.7645615359669125)+((3.1433312852700794)*(1/8.879833171513223)*exp(-0.5*((x-15.507372314680122)/8.879833171513223)^2)))+(0.5*(erf(0.08*(x-20.000000000000004))+1))*((1.0003677574280867)+(-0.0011004779472863187)*x)",  # noqa: E501
            3: "(0.5*(erf(-0.08*(x-21.004122252626765))+1))*((0.6080303297217909)+((9.999999999999996)*(1/15.122210489435163)*exp(-0.5*((x-14.213829801763238)/15.122210489435163)^2)))+(0.5*(erf(0.08*(x-21.004122252626765))+1))*((1.2008443359218373)+(-0.0017550802098248006)*x)",  # noqa: E501
            4: "(0.5*(erf(-0.08*(x-20.000000000000004))+1))*((1.1999999999999997)+((3.1858882233372112)*(1/7.066155941853067)*exp(-0.5*((x-13.130728745354574)/7.066155941853067)^2)))+(0.5*(erf(0.08*(x-20.000000000000004))+1))*((1.55905556834828)+(-0.0018488249685339217)*x)",  # noqa: E501
        },
        "2023postBPix": {
            2: "(0.5*(erf(-0.08*(x-21.89431472888804))+1))*((0.6000000000000001)+((6.726801977956501)*(1/12.346853209201003)*exp(-0.5*((x-15.240479849990233)/12.346853209201003)^2)))+(0.5*(erf(0.08*(x-21.89431472888804))+1))*((1.040432981347636)+(-0.001697363483172575)*x)",  # noqa: E501
            3: "(0.5*(erf(-0.08*(x-20.000000000000004))+1))*((1.1886420740550274)+((0.5314911875265659)*(1/2.774033859961768)*exp(-0.5*((x-10.853660570028293)/2.774033859961768)^2)))+(0.5*(erf(0.08*(x-20.000000000000004))+1))*((1.2809314756262644)+(-0.002539205365672864)*x)",  # noqa: E501
            4: "(0.5*(erf(-0.08*(x-30.94938763988523))+1))*((1.1999999999997815)+((7.928834463409897)*(1/15.904927402578977)*exp(-0.5*((x-14.457545776874468)/15.904927402578977)^2)))+(0.5*(erf(0.08*(x-30.94938763988523))+1))*((1.7099215457047394)+(-0.0034490581369154977)*x)",  # noqa: E501
        },
    }

    # normalization factors per b-tag multiplicity
    # era --> number of central jets --> number of b-tagged jets --> (factor, stat error)
    btag_norms = {
        "2022preEE": {
            2: {
                0: Norm(0.997251140, 0.002422338),
                1: Norm(1.044380589, 0.008200641),
                2: Norm(0.895170017, 0.026916529),
            },
            3: {
                0: Norm(0.990245980, 0.005056013),
                1: Norm(1.089209583, 0.015094569),
                2: Norm(0.939658629, 0.038047498),
            },
            4: {
                0: Norm(0.938042307, 0.010249876),
                1: Norm(1.062744212, 0.028129369),
                2: Norm(1.023759119, 0.066833551),
            },
            5: {
                0: Norm(1.133954255, 0.026400875),
                1: Norm(1.294831981, 0.066763389),
                2: Norm(1.369825379, 0.149507801),
            },
            6: {
                0: Norm(1.364025809, 0.061990316),
                1: Norm(1.708922077, 0.150386411),
                2: Norm(1.833289964, 0.306732068),
            },
        },
        "2022postEE": {
            2: {
                0: Norm(1.0, 0.0),
                1: Norm(1.0, 0.0),
                2: Norm(1.0, 0.0),
            },
            3: {
                0: Norm(1.0, 0.0),
                1: Norm(1.0, 0.0),
                2: Norm(1.0, 0.0),
            },
            4: {
                0: Norm(1.0, 0.0),
                1: Norm(1.0, 0.0),
                2: Norm(1.0, 0.0),
            },
            5: {
                0: Norm(1.0, 0.0),
                1: Norm(1.0, 0.0),
                2: Norm(1.0, 0.0),
            },
            6: {
                0: Norm(1.0, 0.0),
                1: Norm(1.0, 0.0),
                2: Norm(1.0, 0.0),
            },
        },
        "2023preBPix": {
            2: {
                0: Norm(1.0, 0.0),
                1: Norm(1.0, 0.0),
                2: Norm(1.0, 0.0),
            },
            3: {
                0: Norm(1.0, 0.0),
                1: Norm(1.0, 0.0),
                2: Norm(1.0, 0.0),
            },
            4: {
                0: Norm(1.0, 0.0),
                1: Norm(1.0, 0.0),
                2: Norm(1.0, 0.0),
            },
            5: {
                0: Norm(1.0, 0.0),
                1: Norm(1.0, 0.0),
                2: Norm(1.0, 0.0),
            },
            6: {
                0: Norm(1.0, 0.0),
                1: Norm(1.0, 0.0),
                2: Norm(1.0, 0.0),
            },
        },
        "2023postBPix": {
            2: {
                0: Norm(1.0, 0.0),
                1: Norm(1.0, 0.0),
                2: Norm(1.0, 0.0),
            },
            3: {
                0: Norm(1.0, 0.0),
                1: Norm(1.0, 0.0),
                2: Norm(1.0, 0.0),
            },
            4: {
                0: Norm(1.0, 0.0),
                1: Norm(1.0, 0.0),
                2: Norm(1.0, 0.0),
            },
            5: {
                0: Norm(1.0, 0.0),
                1: Norm(1.0, 0.0),
                2: Norm(1.0, 0.0),
            },
            6: {
                0: Norm(1.0, 0.0),
                1: Norm(1.0, 0.0),
                2: Norm(1.0, 0.0),
            },
        },
    }

    # no capping
    # cap_x = lambda s: s
    # cap at 200
    # replace x variable but not x using regex
    cap_x = lambda s: re.sub(r"\bx\b", "min(x,200)", s)
    inf = float("inf")

    def get_factor(era, njets, btag, direction: Literal["nom", "up", "down"]):
        return [(0.0, inf, f"{getattr(btag_norms[era][njets][btag], direction)}*({cap_x(dilep_pt_formulas[era][njets if njets < 5 else 4])})")]  # noqa: E501

    # initialize nested dictionary
    # era --> shift --> number of central jets --> number of b-tagged jets --> scale factor
    dy_weight_data = {}
    for era in dilep_pt_formulas.keys():
        dy_weight_data[era] = {}

        # build nominal dict first
        nom = {
            (njets, njets + 1 if njets < 6 else 101): {
                (btag, btag + 1 if btag < 2 else 101): get_factor(era, njets, btag, "nom")
                for btag in [0, 1, 2]
            }
            for njets in [2, 3, 4, 5, 6]
        }
        dy_weight_data[era]["nom"] = nom

        # consider each b-tag a separate uncertainty source
        for btag in [0, 1, 2]:
            for direction in ["up", "down"]:
                shift_str = f"stat_btag{btag}_{direction}"
                dy_weight_data[era][shift_str] = {}

                bjet_dict = {}
                # shift 0/1/2 btag entry for all njet entries at once
                for jet_key in nom.keys():
                    bjet_dict[jet_key] = law.util.merge_dicts(
                        nom[jet_key],
                        # only update the corresponding btag factor
                        {(btag, btag + 1 if btag < 2 else 101): get_factor(era, jet_key[0], btag, direction)}, deep=True
                    )

                # save shifted btag factor in nested dict
                dy_weight_data[era][shift_str] = bjet_dict

    # create and save the correction set
    cset = cs.CorrectionSet(
        schema_version=2,
        description="Corrections derived for the hh2bbtautau analysis.",
        corrections=[
            create_dy_weight_correction(dy_weight_data),
        ],
    )

    print(cset.model_dump_json(exclude_unset=True))
    with gzip.open("hbt_corrections.json.gz", "wt") as f:
        f.write(cset.model_dump_json(exclude_unset=True))
