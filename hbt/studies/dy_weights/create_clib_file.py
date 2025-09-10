# coding: utf-8

"""
Toy script to explore the creation of a correctionlib file with multiple, mixed dependencies (categories, bin ranges,
and formulas) for the DY weight correction in the hh2bbtautau analysis.
"""

from __future__ import annotations

import re

import correctionlib.schemav2 as cs


# helpers ##############################################################################################################

def step_expr(x: float | int, up: bool, steepness: float | int = 1000) -> str:
    """
    Creates a "step" expression using an error function reach a plateau of 1 at *x*. When *up* is *True*, the step
    increases at *x* (zero-ing all values to the left), otherwise it decreases (zero-ing all values to the right).
    """
    sign = "" if up else "-"
    if x < 0:
        shifted_x = f"(x+{-x})"
    elif x > 0:
        shifted_x = f"(x-{x})"
    else:
        shifted_x = "x"
    return f"0.5*(erf({sign}{steepness}*{shifted_x})+1)"


def expr_in_range(expr: str, lower_bound: float | int, upper_bound: float | int) -> str:
    """
    Multiplies an expression with step functions so that it evaluates to zero outside a range given by *lower_bound* and
    *upper_bound*.
    """
    assert lower_bound < upper_bound, "Lower bound must be smaller than upper bound."
    # distinguish cases
    inf = float("inf")
    bounds = (lower_bound, upper_bound)
    # TODO: fix erf string!
    if bounds == (-inf, inf):
        return expr
    if bounds == (-inf, 0):
        return expr
        # return f"({expr})*{step_expr(0, up=False)}"
    if bounds == (0, inf):
        return expr
        # return f"({expr})*{step_expr(0, up=True)}"
    if lower_bound == -inf:
        return expr
        # return f"({expr})*{step_expr(upper_bound, up=False)}"
    if upper_bound == inf:
        return expr
        # return f"({expr})*{step_expr(lower_bound, up=True)}"
    return expr
    # return f"({expr})*{step_expr(lower_bound, up=True)}*{step_expr(upper_bound, up=False)}"


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
            binning_content = []
            for (min_njet, max_njet), formulas in syst_data.items():
                # create a joined expression for all formulas
                expr = "+".join(
                    expr_in_range(formula, lower_bound, upper_bound)
                    for lower_bound, upper_bound, formula in formulas
                )
                # add formula object to binning content
                binning_content.append(
                    cs.Formula(
                        nodetype="formula",
                        variables=["ptll"],
                        parser="TFormula",
                        expression=expr,
                    ),
                )
            # add a new category item for the jet bins
            njet_edges = sorted(set(sum(map(list, syst_data.keys()), [])))
            era_category_content.append(
                cs.CategoryItem(
                    key=syst,
                    value=cs.Binning(
                        nodetype="binning",
                        input="njets",
                        flow="error",
                        edges=njet_edges,
                        content=binning_content,
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

    formulas = {
        "2022preEE": {
            2: "(0.5*(erf(-0.08*(x-29.35284418720058))+1))*((0.6000000000000001)+((4.13788351736806)*(1/12.68384416493451)*exp(-0.5*((x-16.549864140670717)/12.68384416493451)^2)))+(0.5*(erf(0.08*(x-29.35284418720058))+1))*((0.886372488501618)+(0.0002545066273632575)*x)",
            3: "(0.5*(erf(-0.08*(x-20.000000000006253))+1))*((0.6000000000055947)+((4.684935079638329)*(1/9.749315961972627)*exp(-0.5*((x-14.635668433059768)/9.749315961972627)^2)))+(0.5*(erf(0.08*(x-20.000000000006253))+1))*((0.9937281234059175)+(1.192308108942791e-05)*x)",
            4: "(0.5*(erf(-0.08*(x-24.02708186962039))+1))*((0.7901754443527945)+((5.158864554240971)*(1/8.358857902225806)*exp(-0.5*((x-15.914275456756133)/8.358857902225806)^2)))+(0.5*(erf(0.08*(x-24.02708186962039))+1))*((1.2694392595110884)+(-0.00043892270855478473)*x)",
        },
        "2022postEE": {
            2: "(0.5*(erf(-0.08*(x-32.145006781489))+1))*((0.6000000000000001)+((3.304254342680554)*(1/11.163631796783257)*exp(-0.5*((x-17.145687100915694)/11.163631796783257)^2)))+(0.5*(erf(0.08*(x-32.145006781489))+1))*((0.8850031265877162)+(0.00033950838587539986)*x)",
            3: "(0.5*(erf(-0.08*(x-22.054815668367077))+1))*((0.6000000000000001)+((4.1914827055520565)*(1/11.714555808339519)*exp(-0.5*((x-15.294417995696852)/11.714555808339519)^2)))+(0.5*(erf(0.08*(x-22.054815668367077))+1))*((0.9446099605080525)+(0.0003426496829583549)*x)",
            4: "(0.5*(erf(-0.08*(x-30.583320181625048))+1))*((0.6000000000000001)+((8.818375382937381)*(1/15.782567033596633)*exp(-0.5*((x-16.142725826324664)/15.782567033596633)^2)))+(0.5*(erf(0.08*(x-30.583320181625048))+1))*((1.1834791971110694)+(0.00012342913069317776)*x)",
        },
        "2023preBPix": {
            2: "(0.5*(erf(-0.08*(x-20.000000000000004))+1))*((0.7645615359669125)+((3.1433312852700794)*(1/8.879833171513223)*exp(-0.5*((x-15.507372314680122)/8.879833171513223)^2)))+(0.5*(erf(0.08*(x-20.000000000000004))+1))*((1.0003677574280867)+(-0.0011004779472863187)*x)",
            3: "(0.5*(erf(-0.08*(x-21.004122252626765))+1))*((0.6080303297217909)+((9.999999999999996)*(1/15.122210489435163)*exp(-0.5*((x-14.213829801763238)/15.122210489435163)^2)))+(0.5*(erf(0.08*(x-21.004122252626765))+1))*((1.2008443359218373)+(-0.0017550802098248006)*x)",
            4: "(0.5*(erf(-0.08*(x-20.000000000000004))+1))*((1.1999999999999997)+((3.1858882233372112)*(1/7.066155941853067)*exp(-0.5*((x-13.130728745354574)/7.066155941853067)^2)))+(0.5*(erf(0.08*(x-20.000000000000004))+1))*((1.55905556834828)+(-0.0018488249685339217)*x)",
        },
        "2023postBPix": {
            2: "(0.5*(erf(-0.08*(x-21.89431472888804))+1))*((0.6000000000000001)+((6.726801977956501)*(1/12.346853209201003)*exp(-0.5*((x-15.240479849990233)/12.346853209201003)^2)))+(0.5*(erf(0.08*(x-21.89431472888804))+1))*((1.040432981347636)+(-0.001697363483172575)*x)",
            3: "(0.5*(erf(-0.08*(x-20.000000000000004))+1))*((1.1886420740550274)+((0.5314911875265659)*(1/2.774033859961768)*exp(-0.5*((x-10.853660570028293)/2.774033859961768)^2)))+(0.5*(erf(0.08*(x-20.000000000000004))+1))*((1.2809314756262644)+(-0.002539205365672864)*x)",
            4: "(0.5*(erf(-0.08*(x-30.94938763988523))+1))*((1.1999999999997815)+((7.928834463409897)*(1/15.904927402578977)*exp(-0.5*((x-14.457545776874468)/15.904927402578977)^2)))+(0.5*(erf(0.08*(x-30.94938763988523))+1))*((1.7099215457047394)+(-0.0034490581369154977)*x)",
        },
    }

    normalizations = {
        "2022preEE": {
            4: "0.957005119",
            5: "1.162350131",
            6: "1.374503500",
        },
        "2022postEE": {
            4: "0.960593795",
            5: "1.124956385",
            6: "1.301667345",
        },
        "2023preBPix": {
            4: "0.949943274",
            5: "1.180705634",
            6: "1.444944968",
        },
        "2023postBPix": {
            4: "0.949452852",
            5: "1.191702233",
            6: "1.526680440",
        },
    }

    # no capping
    # cap_x = lambda s: s
    # cap at 200
    # replace x variable but not x using regex
    cap_x = lambda s: re.sub(r"\bx\b", "min(x,200)", s)

    # nested structure of formulas
    # year -> syst -> (min_njet, max_njet) -> [(lower_bound, upper_bound, formula), ...]
    inf = float("inf")
    dy_weight_data = {
        y: {
            "nom": {
                (2, 3): [(0.0, inf, cap_x(formulas[y][2]))],
                (3, 4): [(0.0, inf, cap_x(formulas[y][3]))],
                (4, 5): [(0.0, inf, f"{normalizations[y][4]}*({cap_x(formulas[y][4])})")],
                (5, 6): [(0.0, inf, f"{normalizations[y][5]}*({cap_x(formulas[y][4])})")],
                (6, 101): [(0.0, inf, f"{normalizations[y][6]}*({cap_x(formulas[y][4])})")],
            },
        }
        for y in formulas.keys()
    }
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
