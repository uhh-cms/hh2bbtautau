# coding: utf-8

"""
Histogram hooks for morphing.
"""

from __future__ import annotations

# from collections import defaultdict
from functools import partial

import law
import order as od
# import scinum as sn

from columnflow.util import maybe_import  # , DotDict

np = maybe_import("numpy")
hist = maybe_import("hist")


logger = law.logger.get_logger(__name__)


# define helper functions
def general_read_coupling_values(couplings, poi=["kl", "kt", "c2"]):
    # HOTFIX: change m2p12 to 2p12 for CV in VBFHHto2B2Tau_CV-2p12_C2V-3p87_C3-m5p96_TuneCP5_13p6TeV_madgraph
    if couplings == "kvm2p12_k2v3p87_klm5p96":
        couplings = "kv2p12_k2v3p87_klm5p96"
    coupling_list = couplings.split("_")
    coupling_dict = {}
    for coupling in coupling_list:
        for param in poi:
            if coupling.startswith(param):
                if param == "k2v":
                    coupling_dict[param] = coupling[3:].replace("p", ".").replace("m", "-")
                else:
                    coupling_dict[param] = coupling[2:].replace("p", ".").replace("m", "-")
    # if "c2" in poi:
    #     if "c2" not in coupling_dict.keys():
    #         coupling_dict["c2"] = "0"
    if "kl" in poi:
        if "kl" not in coupling_dict.keys():
            raise ValueError("kl not found in couplings")
    if "kt" in poi:
        if "kt" not in coupling_dict.keys():
            raise ValueError("kt not found in couplings")
    if "kv" in poi:
        if "kv" not in coupling_dict.keys():
            raise ValueError("kv not found in couplings")
    if "k2v" in poi:
        if "k2v" not in coupling_dict.keys():
            raise ValueError("k2v not found in couplings")
    return coupling_dict


def add_hooks(analysis_inst: od.Analysis) -> None:
    """
    Add histogram hooks to a configuration.
    """
    def general_higgs_morphing(
        task,
        hists,
        category_name: str,
        variable_name: str,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1"],
        target_point="kl5_kt1",
        morphing_type="average",
        weights=False,
    ):
        """
        General morphing of Higgs production processes.

        params:
            task: law task
            hists: dict of histograms
            production_channel: str, "vbf" or "ggf", production channel of the Higgs boson
            guidance_points: list of str, guidance points to be used for morphing
            target_point: str, target point to be morphed to
            morphing_type: str, "exact", "average" or "fit", type of morphing to be used
        """

        # check conditions for morphing
        if morphing_type not in ["exact", "average", "fit"]:
            raise ValueError(f"morphing type {morphing_type} not recognized")

        if production_channel not in ["ggf", "vbf"]:
            raise ValueError(f"production channel {production_channel} not recognized")

        if morphing_type == "exact":
            if production_channel == "ggf":
                if len(guidance_points) != 3:
                    raise ValueError("exact morphing requires exactly three guidance points")
            if production_channel == "vbf":
                if len(guidance_points) != 6:
                    raise ValueError("exact morphing requires exactly six guidance points")

        if morphing_type == "average" or morphing_type == "fit":
            if production_channel == "ggf":
                if len(guidance_points) < 4:
                    raise ValueError(f"{morphing_type} morphing requires at least four guidance points")
            if production_channel == "vbf":
                if len(guidance_points) < 7:
                    raise ValueError(f"{morphing_type} morphing requires at least seven guidance points")

        def get_coeffs(guidance_point_dict, parameters_oi):
            """
            Get coefficients from physical parameters for the morphing of the Higgs production processes.
            ggf assumes kl, kt, c2 as ordering in the parameters_oi list
            vbf assumes kv, k2v, kl as ordering in the parameters_oi list
            """
            if parameters_oi == ["kl", "kt", "c2"]:
                # (formula for any kt,kl: (kt**2*kl**2,kt**4,kt**3*kl))
                # assuming parameters_oi = ["kl", "kt", "c2"], c2 is not used in the morphing for now
                coeffs = [
                    (guidance_point_dict[parameters_oi[0]]**2) * (guidance_point_dict[parameters_oi[1]]**2),
                    guidance_point_dict[parameters_oi[1]]**4,
                    guidance_point_dict[parameters_oi[0]] * guidance_point_dict[parameters_oi[1]]**3,
                ]
            elif parameters_oi == ["kv", "k2v", "kl"]:
                # (formula for any kv,k2v,kl: (kv**4,k2v**2,kv**2*kl**2,kv**2*k2v,kv**3*kl,k2v*kv*kl)
                # assuming parameters_oi = ["kv", "k2v", "kl"]
                coeffs = [
                    guidance_point_dict[parameters_oi[0]]**4,
                    guidance_point_dict[parameters_oi[1]]**2,
                    (guidance_point_dict[parameters_oi[0]]**2) * (guidance_point_dict[parameters_oi[2]]**2),
                    (guidance_point_dict[parameters_oi[0]]**2) * guidance_point_dict[parameters_oi[1]],
                    (guidance_point_dict[parameters_oi[0]]**3) * guidance_point_dict[parameters_oi[2]],
                    guidance_point_dict[parameters_oi[0]] * guidance_point_dict[parameters_oi[1]] * guidance_point_dict[parameters_oi[2]],  # noqa
                ]
            else:
                raise ValueError("parameters of interest not recognized, expected [kl, kt, c2] or [kv, k2v, kl]")
            return coeffs

        # define parameters of interest for the production channel, following the order used in get_coeffs
        if production_channel == "ggf":
            parameters_oi = ["kl", "kt", "c2"]  # c2 is not used in the morphing for now
        if production_channel == "vbf":
            parameters_oi = ["kv", "k2v", "kl"]

        # get all guidance points
        guidance_points_list = [general_read_coupling_values(coupling, parameters_oi) for coupling in guidance_points]
        # for coupling in guidance_points:
        #     guidance_points_dict.update(general_read_coupling_values(coupling, parameters_oi))
        guidance_points_list_float = [
            {key: float(sample[key]) for key in sample.keys()} for sample in guidance_points_list
        ]
        # guidance_points_dict_float = {
        #     key: [float(coupling_value) for coupling_value in guidance_points_dict[key]]
        #     for key in guidance_points_dict.keys()
        # }

        # get all model processes

        # TODO: update for multiconfigs hists... Meaning just adding a loop over everything
        # since we won't morph across configs, but only within one config

        # config = task.config_inst
        config = list(hists.keys())[0]
        hists_one_config = hists[config]

        model_procs = []
        for sample in guidance_points:
            model_proc = config.get_process(f"hh_{production_channel}_hbb_htt_{sample}", default=None)
            model_procs.append(model_proc)
        if not all(model_procs):
            raise Exception(f"not all samples available, list of required samples: {guidance_points},"
            f"list of available processes: {model_procs}")

        # verify that the axis order is exactly "category -> shift -> variable"
        # which is needed to insert values at the end
        CAT_AXIS, SHIFT_AXIS, VAR_AXIS = range(3)
        for h in hists_one_config.values():
            # validate axes
            assert len(h.axes) == 3
            assert h.axes[CAT_AXIS].name == "category"
            assert h.axes[SHIFT_AXIS].name == "shift"

        # get model histograms
        model_hists = [h for p, h in hists_one_config.items() if p in model_procs]
        if len(model_hists) != len(guidance_points):
            raise Exception("not all model histograms present, morphing cannot occur")

        # new coefficients for the newly created point
        target_params_oi = general_read_coupling_values(target_point, parameters_oi)
        target_params_oi_float = {key: float(target_params_oi[key]) for key in target_params_oi.keys()}
        new_coefficients = np.array(get_coeffs(target_params_oi_float, parameters_oi))

        # make customized ids
        from law.util import create_hash
        custom_id = create_hash(
            f"{target_point}_{morphing_type}",
            to_int=True,
            l=3,
        )
        if production_channel == "ggf":
            # create the new process for the morphed histogram, TODO: make customized id?
            new_proc = od.Process(
                f"hh_{production_channel}_hbb_htt_{target_point}_{morphing_type}_morphed",
                id=21130 + custom_id,
                label=r"morphed $HH_{ggf} \rightarrow bb\tau\tau$ ",
                # "\n"
                # r"($\kappa_{\lambda}=$" + target_params_oi["kl"] + r", $\kappa_{t}=$" + target_params_oi["kt"] + ")",
            )
        elif production_channel == "vbf":
            # create the new process for the morphed histogram, TODO: make customized id?
            new_proc = od.Process(
                f"hh_{production_channel}_hbb_htt_{target_point}_{morphing_type}_morphed",
                id=21130 + custom_id,
                label=r"morphed $HH_{vbf} \rightarrow bb\tau\tau$ "
                "\n"
                r"($C_{V}=$" + target_params_oi["kv"] + r", $C_{2V}=$" + target_params_oi["k2v"] +
                r", $C_{3}=$" + target_params_oi["kl"] + ")",
            )

        all_categories = config.get_leaf_categories()
        for h in model_hists:
            if len(h.axes[0]) != len(all_categories):
                set_all_categories_names = set([cat.name for cat in all_categories])
                set_h_categories_names = set(list(h.axes[0]))
                missing_categories = list(set_all_categories_names - set_h_categories_names)
                filling_dict = {
                    "category": missing_categories,  # missing category indices to add as axes
                    "shift": h.axes[1][0],  # reuse existing shift to avoid creating a new axis
                    h.axes[2].name: h.axes[2][0][0],  # reuse existing bin edge to avoid creating a new axis
                    # for variables, take the first bin and from that the first (left) edge
                    "weight": 0.0,  # fill with 0
                }
                h.fill(**filling_dict)  # fill with 0, the rest of the axes are filled with 0 to get a regular array

        # create the new hist
        new_hist = model_hists[0].copy().reset()

        # prepare morphing here
        # be careful with the order, the categories can be shuffled around in the different histograms
        # so sort the values by the categories
        model_values = np.array([
            model_hists[i].view().value[np.argsort(model_hists[i].axes[0])] for i in range(len(guidance_points))
        ])

        model_variances = np.array([
            model_hists[i].view().variance[np.argsort(model_hists[i].axes[0])] for i in range(len(guidance_points))
        ])

        original_hist_shape = new_hist.view().value.shape

        if morphing_type == "exact" or morphing_type == "average":
            # build guidance matrix from guidance points
            guidance_matrix = np.array([
                get_coeffs(guidance_points_list_float[i], parameters_oi)
                for i in range(len(guidance_points))
            ])

            if morphing_type == "exact":
                # inverse guidance matrix
                inv_guidance_matrix = np.linalg.inv(guidance_matrix)
            elif morphing_type == "average":
                # pseudo inverse guidance matrix
                inv_guidance_matrix = np.linalg.pinv(guidance_matrix)

            # morphing
            # morphed values
            morphed_values = np.matmul(
                np.matmul(new_coefficients, inv_guidance_matrix),
                model_values.reshape(len(guidance_points), -1),
            ).reshape(original_hist_shape)

            # morphed values and variances, using covariance matrix
            reshaped_model_variances = model_variances.reshape(len(guidance_points), -1)
            covariances = np.array([
                np.diag(reshaped_model_variances[:, i]) for i in range(reshaped_model_variances.shape[1])
            ])
            sigma_v_vector = np.matmul(
                np.matmul(
                    inv_guidance_matrix,
                    covariances.T,
                ).T,
                inv_guidance_matrix.T,
            )

            morphed_variances = np.matmul(
                new_coefficients,
                np.matmul(
                    sigma_v_vector,
                    new_coefficients,
                ).T,
            ).reshape(original_hist_shape)

            # # alternative method for morphing, using the actual equation, even for non invertible matrices:
            # # taken from blobel statistics book
            # # leads to exact same results as above
            # # (C^T * C) v = C^T sigma
            # # v = (C^T * C)^-1 * C^T sigma
            # inv_guidance_matrix_v2 = np.linalg.inv(np.matmul(guidance_matrix.T, guidance_matrix))
            # factor_v = np.matmul(inv_guidance_matrix_v2, guidance_matrix.T)
            # morphed_values_v2 = np.matmul(
            #     new_coefficients,
            #     np.matmul(
            #         factor_v,
            #         model_values.reshape(len(guidance_points), -1),
            #     ),
            # ).reshape(original_hist_shape)

            # sigma_v_vector_v2 = np.matmul(
            #     np.matmul(
            #         factor_v,
            #         covariances.T,
            #     ).T,
            #     factor_v.T,
            # )

            # morphed_variances_v2 = np.matmul(
            #     new_coefficients,
            #     np.matmul(
            #         sigma_v_vector_v2,
            #         new_coefficients,
            #     ).T,
            # ).reshape(original_hist_shape)

            if weights:
                # morphing with weights matrix, needs equation from blobel statistics book
                # actual equation also possible with weights
                # v = (C^T * W * C)^-1 * C^T * W * sigma

                # define weights
                weights = np.array([
                    np.diag(1 / reshaped_model_variances[:, i]) for i in range(reshaped_model_variances.shape[1])
                ])

                inv_guidance_matrix_v3 = np.linalg.inv(
                    np.matmul(
                        np.matmul(guidance_matrix.T, weights),
                        guidance_matrix,
                    ),
                )

                factor_v_2 = np.matmul(
                    np.matmul(inv_guidance_matrix_v3, guidance_matrix.T),
                    weights,
                )

                # at this point, matrix multiplication becomes difficult due to extended dimensionality
                # we want to multiply the matrices for each bin separately

                # # method number 1: custom multiplication by hand in loop
                # vector_v_3 = np.array([
                #     np.matmul(factor_v_2[i], model_values.reshape(len(guidance_points), -1)[:, i]) for i in range(len(factor_v_2))  # noqa
                # ])

                # morphed_values_with_weights = np.matmul(
                #     new_coefficients,
                #     vector_v_3.T,
                # ).reshape(original_hist_shape)

                # method number 2: tensor multiplication, then masking the additional dimensions
                tensordot_res = np.tensordot(
                    factor_v_2,
                    model_values.reshape(len(guidance_points), -1),
                    axes=1,
                )
                # mask creation
                mask = np.zeros_like(tensordot_res, dtype=bool)
                for i in range(len(factor_v_2)):
                    mask[i, :, i] = True

                vector_v_3_v2 = tensordot_res[mask].reshape(len(factor_v_2), -1)

                morphed_values_with_weights = np.matmul(
                    new_coefficients,
                    vector_v_3_v2.T,
                ).reshape(original_hist_shape)

                # calculate the morphed variances with weights

                # start with the first multiplication of the covariances with the prefactor of v
                tensordot_res_variances_part1 = np.tensordot(
                    factor_v_2,
                    covariances.T,
                    axes=1,
                )

                mask_variances_part1 = np.zeros_like(tensordot_res_variances_part1, dtype=bool)
                for i in range(len(factor_v_2)):
                    mask_variances_part1[i, :, :, i] = True

                sigma_vector_v_3_part1 = tensordot_res_variances_part1[mask_variances_part1].reshape(
                    len(factor_v_2),
                    len(new_coefficients),
                    -1,
                )

                # Then do the second multiplication of the covariances with the prefactor of v

                tensordot_res_variances_part2 = np.tensordot(
                    sigma_vector_v_3_part1,
                    factor_v_2.T,
                    axes=1,
                )

                mask_variances_part2 = np.zeros_like(tensordot_res_variances_part2, dtype=bool)
                for i in range(len(factor_v_2)):
                    mask_variances_part2[i, :, :, i] = True

                sigma_v_vector_v3 = tensordot_res_variances_part2[mask_variances_part2].reshape(
                    len(factor_v_2),
                    len(new_coefficients),
                    len(new_coefficients),
                )

                # finally, do the last multiplication with the new coefficients to get the morphed variances
                morphed_variances_with_weights = np.matmul(
                    new_coefficients,
                    np.matmul(
                        sigma_v_vector_v3,
                        new_coefficients,
                    ).T,
                ).reshape(original_hist_shape)

                morphed_values = morphed_values_with_weights
                morphed_variances = morphed_variances_with_weights

        if morphing_type == "fit":
            # create function and define params
            from scipy.optimize import curve_fit
            x = []
            for i in range(len(guidance_points)):
                x.append([float(guidance_points_list[i][key]) for key in guidance_points_list[i].keys()])
            x = np.array(x)

            if production_channel == "ggf":
                def LO_fit_function(values, a, b, c):
                    kl, kt = values
                    return a * kt**2 * kl**2 + b * kt**4 + c * kl * kt**3

                def LO_fit_variances(values, a, b, c):
                    kl, kt = values
                    return a * (kt**2 * kl**2)**2 + b * (kt**4)**2 + c * (kl * kt**3)**2

                def LO_primitive_funtion(values, a, b, c):
                    kl, kt = values
                    return a * kt**3 * kl**3 / 9 + b * kt**5 / 5 + c * kl**2 * kt**4 / 8

            elif production_channel == "vbf":
                def LO_fit_function(values, a, b, c, d, e, f):
                    kv, k2v, kl = values
                    return a * kv**4 + b * k2v**2 + c * (kv**2) * (kl**2) + d * (kv**2) * k2v + e * (kv**3) * kl + f * k2v * kv * kl  # noqa

                def LO_fit_variances(values, a, b, c, d, e, f):
                    kv, k2v, kl = values
                    return a * (kv**4)**2 + b * (k2v**2)**2 + c * (kv**2 * kl**2)**2 + d * (kv**2 * k2v)**2 + e * (kv**3 * kl)**2 + f * (k2v * kv * kl)**2  # noqa

                def LO_primitive_funtion(values, a, b, c, d, e, f):
                    kv, k2v, kl = values

                    return a / 5 * kv**5 + b / 3 * k2v**3 + c / 9 * (kv**3) * (kl**3) + d / 6 * (kv**3) * (k2v**2) + e / 8 * (kv**4) * (kl**2) + f / 8 * (k2v**2) * (kv**2) * (kl**2)  # noqa

            morphed_values = np.zeros(model_values[0].shape)
            morphed_variances = np.zeros(model_values[0].shape)

            pcovs = []
            # minuit_pcovs = []
            # chi2_infos = {}  # should include chi2 value, ndf, p-value, cov and fit status

            # loop over bins and calculate the morphed value
            for cat_ in range(len(model_hists[0].axes[0].centers)):
                for shift in range(len(model_hists[0].axes[1].centers)):
                    for bin_ in range(len(model_hists[0].axes[2].centers)):
                        # fit the bin values to the function
                        y = model_values[:, cat_, shift, bin_]
                        variances = model_variances[:, cat_, shift, bin_]
                        if production_channel == "ggf":
                            p0_fit = [0, 0, 0]
                        elif production_channel == "vbf":
                            p0_fit = [0, 0, 0, 0, 0, 0]

                        # y = np.array([model_hists[i].view().value[bin] for i in range(len(guidance_points))])
                        popt, pcov, infodict_, mesg_, ier_ = curve_fit(
                            LO_fit_function,
                            x.T,
                            y,
                            sigma=np.sqrt(variances),
                            absolute_sigma=True,
                            p0=p0_fit,
                            full_output=True,
                        )

                        # # alternative least square fitting with iminuit
                        # from iminuit import Minuit
                        # # we also need a cost function to fit and import the LeastSquares function
                        # from iminuit.cost import LeastSquares
                        # least_squares = LeastSquares(x.T, y, np.sqrt(variances), LO_fit_function)
                        # if production_channel == "ggf":
                        #     initial_params_dict = {"a": 0, "b": 0, "c": 0}
                        # elif production_channel == "vbf":
                        #     initial_params_dict = {"a": 0, "b": 0, "c": 0, "d": 0, "e": 0, "f": 0}
                        # m = Minuit(least_squares, **initial_params_dict)
                        # # m.errordef = Minuit.LIKELIHOOD
                        # m.migrad()  # finds minimum of least_squares function
                        # m.hesse()  # accurately computes uncertainties
                        # fitted_params = np.array(m.values)
                        # # fitted_param_errors = list(m.errors)
                        # fitted_cov = np.array(m.covariance)

                        # TODO : WIP
                        # # alternative fitting with likelihood
                        # from iminuit import Minuit
                        # # we also need a cost function to fit and import the LeastSquares function
                        # from iminuit.cost import ExtendedUnbinnedNLL

                        # def general_pdf(x, *args, channel=production_channel):
                        #     # from PDG Higgs section:
                        #     # https://pdg.lbl.gov/2024/web/viewer.html?file=../reviews/rpp2024-rev-higgs-boson.pdf
                        #     # kt is measured at 1.01 +- 0.11 by CMS and 0.94 +- 0.11 by ATLAS
                        #     # Therefore, we will constrain it at the first integer above 5\sigma: [0, 2]
                        #     #
                        #     # kw is measured at 1.02 +- 0.08 by CMS and 1.05 +- 0.06 by ATLAS
                        #     # kz is measured at 1.04 +- 0.07 by CMS and 0.99 +- 0.06 by ATLAS
                        #     # as the kv variations in the samples are much larger, we will constrain it to [-2.5,2]
                        #     # to fit all samples inside the range
                        #     #
                        #     # kl is constrained to [-1.2,6.5] by CMS and [-0.6, 6.6] by ATLAS at 95% CL in HH analyses
                        #     # (ATLAS has [-0.4, 6.3] for single Higgs analyses)
                        #     # as the kl variations in the samples are much larger, we will constrain it to [-20,15]
                        #     # to fit all samples inside the range
                        #     #
                        #     # k2v is constrained to [-0.1, 2.2] by CMS and [-0.1,2.0] by ATLAS at 95% CL
                        #     # To emulate a 5\sigma CL for a gaussian, we will constrain it to [-3,4]
                        #     # which fits all samples inside the range

                        #     if channel == "ggf":
                        #         kl, kt = x
                        #         c_kl = [-20, 15]
                        #         c_kt = [0, 2]
                        #         if kl < c_kl[0] or kl > c_kl[1] or kt < c_kt[0] or kt > c_kt[1]:
                        #             return 0
                        #         else:
                        #             integral_1 = LO_primitive_funtion([c_kl[1], c_kt[1]], *args)
                        #             integral_2 = LO_primitive_funtion([c_kl[0], c_kt[1]], *args)
                        #             integral_3 = LO_primitive_funtion([c_kl[1], c_kt[0]], *args)
                        #             integral_4 = LO_primitive_funtion([c_kl[0], c_kt[0]], *args)
                        #             integral = integral_1 - integral_2 - integral_3 + integral_4
                        #     elif channel == "vbf":
                        #         kv, k2v, kl = x
                        #         c_kv = [-2.5, 2]
                        #         c_k2v = [-3, 4]
                        #         c_kl = [-20, 15]
                        #         if kv < c_kv[0] or kv > c_kv[1] or k2v < c_k2v[0] or k2v > c_k2v[1] or kl < c_kl[0] or kl > c_kl[1]:  # noqa
                        #             return 0
                        #         else:
                        #             integral_1 = LO_primitive_funtion([c_kv[1], c_k2v[1], c_kl[1]], *args)
                        #             integral_2 = LO_primitive_funtion([c_kv[1], c_k2v[1], c_kl[0]], *args)
                        #             integral_3 = LO_primitive_funtion([c_kv[1], c_k2v[0], c_kl[1]], *args)
                        #             integral_4 = LO_primitive_funtion([c_kv[0], c_k2v[1], c_kl[1]], *args)
                        #             integral_5 = LO_primitive_funtion([c_kv[0], c_k2v[0], c_kl[1]], *args)
                        #             integral_6 = LO_primitive_funtion([c_kv[0], c_k2v[1], c_kl[0]], *args)
                        #             integral_7 = LO_primitive_funtion([c_kv[1], c_k2v[0], c_kl[0]], *args)
                        #             integral_8 = LO_primitive_funtion([c_kv[0], c_k2v[0], c_kl[0]], *args)
                        #             integral = integral_1 - integral_2 - integral_3 - integral_4 + integral_5 + integral_6 + integral_7 - integral_8  # noqa
                        #     return LO_fit_function(x, *args) / integral

                        # pdf = partial(general_pdf, channel=production_channel)

                        # likelihood = ExtendedUnbinnedNLL(y, pdf)
                        # # from iminuit.cost import LeastSquares
                        # # least_squares = LeastSquares(x.T, y, np.sqrt(variances), LO_fit_function)
                        # m = Minuit(likelihood, a=0, b=0, c=0)
                        # # m.errordef = Minuit.LIKELIHOOD
                        # m.migrad()  # finds minimum of least_squares function
                        # m.hesse()  # accurately computes uncertainties
                        # popt = list(m.values)
                        # from IPython import embed; embed(header="fitting, check errors")
                        # # pcov = list(m.errors)

                        # calculate the morphed value
                        target_param_values = [float(target_params_oi[key]) for key in target_params_oi.keys()]
                        morphed_value = LO_fit_function(target_param_values, *popt)
                        fit_variance = np.diag(pcov)
                        morphed_variance = LO_fit_variances(target_param_values, *fit_variance)

                        morphed_values[cat_, shift, bin_] = morphed_value
                        morphed_variances[cat_, shift, bin_] = morphed_variance
                        # if cat_ == 18 and bin_ == 5:
                        #     print(f"cat: {cat_}, shift: {shift}, bin: {bin_}")
                        #     print(f"morphed_value: {morphed_value}")
                        #     print(f"morphed_variance: {morphed_variance}")
                        #     from IPython import embed; embed()
                        pcovs.append(pcov)
                        # minuit_pcovs.append(fitted_cov)

            pcovs = np.array(pcovs)
            # minuit_pcovs = np.array(minuit_pcovs)
            morphed_variances = np.matmul(
                new_coefficients,
                np.matmul(
                    pcovs,
                    new_coefficients,
                ).T,
            ).reshape(original_hist_shape)

            # TODO: check why uncertainties fit and weights method are not the same
            # from IPython import embed; embed(header="fitting")

        # reshape the values to the correct categorization
        morphed_values_correct_categorization = morphed_values[np.argsort(np.argsort(model_hists[0].axes[0]))]
        morphed_variances_correct_categorization = morphed_variances[np.argsort(np.argsort(model_hists[0].axes[0]))]

        # insert the new hist into the hists dict
        new_proc.x.guidance_points_procs = model_procs
        new_proc.x.morphing_type = morphing_type
        new_proc.x.target_point = target_point
        new_proc.x.production_channel = production_channel
        new_proc.x.guidance_points_values = guidance_points_list_float

        # difficult to use for plotting since the categories to be added are not known
        # so maybe add as a separate histogram?
        # new_proc.x.chi2_infos = chi2_infos

        hists[config][new_proc] = new_hist

        # insert values into the new histogram
        hists[config][new_proc].view().value = morphed_values_correct_categorization
        hists[config][new_proc].view().variance = morphed_variances_correct_categorization

        return hists

    # add the hooks
    analysis_inst.x.hist_hooks.hh_ggf_exact_morphing_kl0_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl0_kt1",
        morphing_type="exact",
    )
    analysis_inst.x.hist_hooks.hh_ggf_exact_morphing_kl1_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl1_kt1",
        morphing_type="exact",
    )
    analysis_inst.x.hist_hooks.hh_ggf_exact_morphing_kl2_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1"],
        target_point="kl2_kt1",
        morphing_type="exact",
    )
    analysis_inst.x.hist_hooks.hh_ggf_exact_morphing_kl2p45_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl5_kt1"],
        target_point="kl2p45_kt1",
        morphing_type="exact",
    )
    analysis_inst.x.hist_hooks.hh_ggf_exact_morphing_kl3_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1"],
        target_point="kl3_kt1",
        morphing_type="exact",
    )
    analysis_inst.x.hist_hooks.hh_ggf_exact_morphing_kl4_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1"],
        target_point="kl4_kt1",
        morphing_type="exact",
    )
    analysis_inst.x.hist_hooks.hh_ggf_exact_morphing_kl5_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1"],
        target_point="kl5_kt1",
        morphing_type="exact",
    )

    analysis_inst.x.hist_hooks.hh_ggf_average_morphing_kl0_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl0_kt1",
        morphing_type="average",
    )
    analysis_inst.x.hist_hooks.hh_ggf_average_morphing_kl1_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl1_kt1",
        morphing_type="average",
    )
    analysis_inst.x.hist_hooks.hh_ggf_average_morphing_kl2_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl2_kt1",
        morphing_type="average",
    )
    analysis_inst.x.hist_hooks.hh_ggf_average_morphing_kl2p45_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl2p45_kt1",
        morphing_type="average",
    )
    analysis_inst.x.hist_hooks.hh_ggf_average_morphing_kl3_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl3_kt1",
        morphing_type="average",
    )
    analysis_inst.x.hist_hooks.hh_ggf_average_morphing_kl4_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl4_kt1",
        morphing_type="average",
    )
    analysis_inst.x.hist_hooks.hh_ggf_average_morphing_kl5_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl5_kt1",
        morphing_type="average",
    )
    analysis_inst.x.hist_hooks.hh_ggf_average_morphing_klm1p7_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="klm1p7_kt1",
        morphing_type="average",
    )
    analysis_inst.x.hist_hooks.hh_ggf_average_morphing_kl8p7_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl8p7_kt1",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_ggf_weight_morphing_kl0_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl0_kt1",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_ggf_weight_morphing_kl1_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl1_kt1",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_ggf_weight_morphing_kl2_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl2_kt1",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_ggf_weight_morphing_kl2p45_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl2p45_kt1",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_ggf_weight_morphing_kl3_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl3_kt1",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_ggf_weight_morphing_kl4_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl4_kt1",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_ggf_weight_morphing_kl5_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl5_kt1",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_ggf_fit_morphing_kl0_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl0_kt1",
        morphing_type="fit",
    )
    analysis_inst.x.hist_hooks.hh_ggf_fit_morphing_kl1_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl1_kt1",
        morphing_type="fit",
    )
    analysis_inst.x.hist_hooks.hh_ggf_fit_morphing_kl2_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl2_kt1",
        morphing_type="fit",
    )
    analysis_inst.x.hist_hooks.hh_ggf_fit_morphing_kl2p45_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl2p45_kt1",
        morphing_type="fit",
    )
    analysis_inst.x.hist_hooks.hh_ggf_fit_morphing_kl3_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl3_kt1",
        morphing_type="fit",
    )
    analysis_inst.x.hist_hooks.hh_ggf_fit_morphing_kl4_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl4_kt1",
        morphing_type="fit",
    )
    analysis_inst.x.hist_hooks.hh_ggf_fit_morphing_kl5_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl5_kt1",
        morphing_type="fit",
    )
    analysis_inst.x.hist_hooks.hh_ggf_fit_morphing_klm1p7_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="klm1p7_kt1",
        morphing_type="fit",
    )
    analysis_inst.x.hist_hooks.hh_ggf_fit_morphing_kl8p7_kt1 = partial(
        general_higgs_morphing,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
        target_point="kl8p7_kt1",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_exact_morphing_kv1_k2v1_kl2 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm1p21_k2v1p94_klm0p94",  # kvm0p012_k2v0p03_kl10p2
        ],
        target_point="kv1_k2v1_kl2",
        morphing_type="exact",
    )

    analysis_inst.x.hist_hooks.hh_vbf_exact_morphing_kv1_k2v1_kl1 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm1p21_k2v1p94_klm0p94", "kvm0p012_k2v0p03_kl10p2",
            "kvm0p962_k2v0p959_klm1p43",
        ],
        target_point="kv1_k2v1_kl1",
        morphing_type="exact",
    )

    # exact morphing using first 6 points except if the one to be morphed is inside, then use other first 6 points
    analysis_inst.x.hist_hooks.hh_vbf_exact_morphing_kv1_k2v0_kl1 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2",
        ],
        target_point="kv1_k2v0_kl1",
        morphing_type="exact",
    )

    analysis_inst.x.hist_hooks.hh_vbf_exact_morphing_kv1_k2v1_kl2 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2",
        ],
        target_point="kv1_k2v1_kl2",
        morphing_type="exact",
    )

    analysis_inst.x.hist_hooks.hh_vbf_exact_morphing_kv1_k2v2_kl1 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2",
        ],
        target_point="kv1_k2v2_kl1",
        morphing_type="exact",
    )

    analysis_inst.x.hist_hooks.hh_vbf_exact_morphing_kv1p74_k2v1p37_kl14p4 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2",
        ],
        target_point="kv1p74_k2v1p37_kl14p4",
        morphing_type="exact",
    )

    analysis_inst.x.hist_hooks.hh_vbf_exact_morphing_kvm0p758_k2v1p44_klm19p3 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4",
            "kvm0p012_k2v0p03_kl10p2",
        ],
        target_point="kvm0p758_k2v1p44_klm19p3",
        morphing_type="exact",
    )

    analysis_inst.x.hist_hooks.hh_vbf_exact_morphing_kvm0p012_k2v0p03_kl10p2 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
        ],
        target_point="kvm0p012_k2v0p03_kl10p2",
        morphing_type="exact",
    )

    analysis_inst.x.hist_hooks.hh_vbf_exact_morphing_kvm0p962_k2v0p959_klm1p43 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
        ],
        target_point="kvm0p962_k2v0p959_klm1p43",
        morphing_type="exact",
    )

    analysis_inst.x.hist_hooks.hh_vbf_exact_morphing_kvm1p21_k2v1p94_klm0p94 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
        ],
        target_point="kvm1p21_k2v1p94_klm0p94",
        morphing_type="exact",
    )

    analysis_inst.x.hist_hooks.hh_vbf_exact_morphing_kvm1p6_k2v2p72_klm1p36 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
        ],
        target_point="kvm1p6_k2v2p72_klm1p36",
        morphing_type="exact",
    )

    analysis_inst.x.hist_hooks.hh_vbf_exact_morphing_kvm1p83_k2v3p57_klm3p39 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
        ],
        target_point="kvm1p83_k2v3p57_klm3p39",
        morphing_type="exact",
    )

    analysis_inst.x.hist_hooks.hh_vbf_exact_morphing_kv2p12_k2v3p87_klm5p96 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
        ],
        target_point="kv2p12_k2v3p87_klm5p96",
        morphing_type="exact",
    )

    # all points
    # analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1_k2v1_kl5 = partial(
    #     general_higgs_morphing,
    #     production_channel="vbf",
    #     guidance_points=[
    #         "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
    #         "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
    #         "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
    #         "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
    #         "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96"
    #     ],
    #     target_point="kv1_k2v1_kl5",
    #     morphing_type="fit",
    # )

    ###############################################################################################################
    ###############################################################################################################
    ###############################################################################################################

    # all points except SM
    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1_k2v1_kl5_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl5",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1_k2v1_kl5 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl5",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1_k2v1_kl1_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl1",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1_k2v1_kl1 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl1",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kv1_k2v1_kl1_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl1",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kv1_k2v1_kl1 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl1",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv1_k2v1_kl5_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl5",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv1_k2v1_kl5 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl5",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv1_k2v1_kl1_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl1",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv1_k2v1_kl1 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl1",
        morphing_type="fit",
    )

    # all points except the one to be morphed
    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1_k2v0_kl1_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v0_kl1",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1_k2v0_kl1 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v0_kl1",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kv1_k2v0_kl1_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v0_kl1",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kv1_k2v0_kl1 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v0_kl1",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv1_k2v0_kl1_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v0_kl1",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv1_k2v0_kl1 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v0_kl1",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1_k2v1_kl2_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl2",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kv1_k2v1_kl2_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl2",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv1_k2v1_kl2_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl2",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1_k2v2_kl1_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v2_kl1",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kv1_k2v2_kl1_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v2_kl1",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv1_k2v2_kl1_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v2_kl1",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1p74_k2v1p37_kl14p4_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1p74_k2v1p37_kl14p4",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1p74_k2v1p37_kl14p4 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1p74_k2v1p37_kl14p4",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kv1p74_k2v1p37_kl14p4_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1p74_k2v1p37_kl14p4",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kv1p74_k2v1p37_kl14p4 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1p74_k2v1p37_kl14p4",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv1p74_k2v1p37_kl14p4_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1p74_k2v1p37_kl14p4",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv1p74_k2v1p37_kl14p4 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1p74_k2v1p37_kl14p4",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kvm0p758_k2v1p44_klm19p3_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm0p758_k2v1p44_klm19p3",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kvm0p758_k2v1p44_klm19p3 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm0p758_k2v1p44_klm19p3",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kvm0p758_k2v1p44_klm19p3_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm0p758_k2v1p44_klm19p3",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kvm0p758_k2v1p44_klm19p3 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm0p758_k2v1p44_klm19p3",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kvm0p758_k2v1p44_klm19p3_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm0p758_k2v1p44_klm19p3",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kvm0p758_k2v1p44_klm19p3 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm0p758_k2v1p44_klm19p3",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kvm0p012_k2v0p03_kl10p2_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm0p012_k2v0p03_kl10p2",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kvm0p012_k2v0p03_kl10p2 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm0p012_k2v0p03_kl10p2",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kvm0p012_k2v0p03_kl10p2_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm0p012_k2v0p03_kl10p2",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kvm0p012_k2v0p03_kl10p2 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm0p012_k2v0p03_kl10p2",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kvm0p012_k2v0p03_kl10p2_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm0p012_k2v0p03_kl10p2",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kvm0p012_k2v0p03_kl10p2 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm0p012_k2v0p03_kl10p2",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kvm0p962_k2v0p959_klm1p43_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm0p962_k2v0p959_klm1p43",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kvm0p962_k2v0p959_klm1p43 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm0p962_k2v0p959_klm1p43",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kvm0p962_k2v0p959_klm1p43_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm0p962_k2v0p959_klm1p43",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kvm0p962_k2v0p959_klm1p43 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm0p962_k2v0p959_klm1p43",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kvm0p962_k2v0p959_klm1p43_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm0p962_k2v0p959_klm1p43",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kvm0p962_k2v0p959_klm1p43 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm0p962_k2v0p959_klm1p43",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kvm1p21_k2v1p94_klm0p94_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p21_k2v1p94_klm0p94",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kvm1p21_k2v1p94_klm0p94 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p21_k2v1p94_klm0p94",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kvm1p21_k2v1p94_klm0p94_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p21_k2v1p94_klm0p94",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kvm1p21_k2v1p94_klm0p94 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p21_k2v1p94_klm0p94",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kvm1p21_k2v1p94_klm0p94_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p21_k2v1p94_klm0p94",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kvm1p21_k2v1p94_klm0p94 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p21_k2v1p94_klm0p94",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kvm1p6_k2v2p72_klm1p36_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p6_k2v2p72_klm1p36",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kvm1p6_k2v2p72_klm1p36 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p6_k2v2p72_klm1p36",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kvm1p6_k2v2p72_klm1p36_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p6_k2v2p72_klm1p36",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kvm1p6_k2v2p72_klm1p36 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p6_k2v2p72_klm1p36",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kvm1p6_k2v2p72_klm1p36_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p6_k2v2p72_klm1p36",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kvm1p6_k2v2p72_klm1p36 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p6_k2v2p72_klm1p36",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kvm1p83_k2v3p57_klm3p39_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p83_k2v3p57_klm3p39",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kvm1p83_k2v3p57_klm3p39 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p83_k2v3p57_klm3p39",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kvm1p83_k2v3p57_klm3p39_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p83_k2v3p57_klm3p39",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kvm1p83_k2v3p57_klm3p39 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p83_k2v3p57_klm3p39",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kvm1p83_k2v3p57_klm3p39_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p83_k2v3p57_klm3p39",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kvm1p83_k2v3p57_klm3p39 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p83_k2v3p57_klm3p39",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv2p12_k2v3p87_klm5p96_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kv2p12_k2v3p87_klm5p96",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv2p12_k2v3p87_klm5p96 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kv2p12_k2v3p87_klm5p96",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kv2p12_k2v3p87_klm5p96_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kv2p12_k2v3p87_klm5p96",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kv2p12_k2v3p87_klm5p96 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kv2p12_k2v3p87_klm5p96",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv2p12_k2v3p87_klm5p96_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kv2p12_k2v3p87_klm5p96",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv2p12_k2v3p87_klm5p96 = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kv2p12_k2v3p87_klm5p96",
        morphing_type="fit",
    )

    ###############################################################################################################
    ###############################################################################################################
    ###############################################################################################################

    # all points
    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1_k2v1_kl0_all_points_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl0",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1_k2v1_kl0_all_points = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl0",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv1_k2v1_kl0_all_points_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl0",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv1_k2v1_kl0_all_points = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl0",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1_k2v1_kl1_all_points_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl1",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1_k2v1_kl1_all_points = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl1",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kv1_k2v1_kl1_all_points_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl1",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kv1_k2v1_kl1_all_points = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl1",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv1_k2v1_kl1_all_points_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl1",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv1_k2v1_kl1_all_points = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl1",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1_k2v1_kl5_all_points_2022_pre = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl5",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1_k2v1_kl5_all_points = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv1_k2v1_kl5",
        morphing_type="average",
    )

    ###############################################################################################################
    ###############################################################################################################
    ###############################################################################################################

    # all points except the one to be morphed and kv2p12_k2v3p87_klm5p96
    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1_k2v1_kl1_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kv1_k2v1_kl1",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kv1_k2v1_kl1_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kv1_k2v1_kl1",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv1_k2v1_kl1_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kv1_k2v1_kl1",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1_k2v0_kl1_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kv1_k2v0_kl1",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kv1_k2v0_kl1_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kv1_k2v0_kl1",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv1_k2v0_kl1_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kv1_k2v0_kl1",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1_k2v1_kl2_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kv1_k2v1_kl2",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kv1_k2v1_kl2_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kv1_k2v1_kl2",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv1_k2v1_kl2_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kv1_k2v1_kl2",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1_k2v2_kl1_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kv1_k2v2_kl1",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kv1_k2v2_kl1_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kv1_k2v2_kl1",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv1_k2v2_kl1_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kv1_k2v2_kl1",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kv1p74_k2v1p37_kl14p4_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kv1p74_k2v1p37_kl14p4",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kv1p74_k2v1p37_kl14p4_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kv1p74_k2v1p37_kl14p4",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv1p74_k2v1p37_kl14p4_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kv1p74_k2v1p37_kl14p4",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kvm0p758_k2v1p44_klm19p3_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kvm0p758_k2v1p44_klm19p3",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kvm0p758_k2v1p44_klm19p3_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kvm0p758_k2v1p44_klm19p3",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kvm0p758_k2v1p44_klm19p3_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kvm0p758_k2v1p44_klm19p3",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kvm0p012_k2v0p03_kl10p2_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kvm0p012_k2v0p03_kl10p2",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kvm0p012_k2v0p03_kl10p2_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kvm0p012_k2v0p03_kl10p2",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kvm0p012_k2v0p03_kl10p2_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kvm0p012_k2v0p03_kl10p2",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kvm0p962_k2v0p959_klm1p43_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kvm0p962_k2v0p959_klm1p43",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kvm0p962_k2v0p959_klm1p43_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kvm0p962_k2v0p959_klm1p43",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kvm0p962_k2v0p959_klm1p43_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kvm0p962_k2v0p959_klm1p43",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kvm1p21_k2v1p94_klm0p94_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kvm1p21_k2v1p94_klm0p94",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kvm1p21_k2v1p94_klm0p94_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kvm1p21_k2v1p94_klm0p94",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kvm1p21_k2v1p94_klm0p94_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kvm1p21_k2v1p94_klm0p94",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kvm1p6_k2v2p72_klm1p36_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kvm1p6_k2v2p72_klm1p36",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kvm1p6_k2v2p72_klm1p36_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kvm1p6_k2v2p72_klm1p36",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kvm1p6_k2v2p72_klm1p36_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94",
            "kvm1p83_k2v3p57_klm3p39",
        ],
        target_point="kvm1p6_k2v2p72_klm1p36",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_average_morphing_kvm1p83_k2v3p57_klm3p39_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p83_k2v3p57_klm3p39",
        morphing_type="average",
    )

    analysis_inst.x.hist_hooks.hh_vbf_weight_morphing_kvm1p83_k2v3p57_klm3p39_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p83_k2v3p57_klm3p39",
        morphing_type="average",
        weights=True,
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kvm1p83_k2v3p57_klm3p39_test = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p83_k2v3p57_klm3p39",
        morphing_type="fit",
    )

    ###############################################################################
    # basis tests
    ###############################################################################
    basis_resolved = [
        "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        "kvm0p012_k2v0p03_kl10p2", "kvm1p21_k2v1p94_klm0p94",
    ]
    basis_boosted = [
        "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1p74_k2v1p37_kl14p4", "kvm0p962_k2v0p959_klm1p43",
        "kvm0p758_k2v1p44_klm19p3", "kvm1p6_k2v2p72_klm1p36",
    ]
    analysis_inst.x.hist_hooks.hh_vbf_exact_morphing_kvm1p6_k2v2p72_klm1p36_basis_resolved = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=basis_resolved,
        target_point="kvm1p6_k2v2p72_klm1p36",
        morphing_type="exact",
    )

    analysis_inst.x.hist_hooks.hh_vbf_exact_morphing_kvm1p21_k2v1p94_klm0p94_basis_boosted = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=basis_boosted,
        target_point="kvm1p21_k2v1p94_klm0p94",
        morphing_type="exact",
    )

    analysis_inst.x.hist_hooks.hh_vbf_exact_morphing_kv2p12_k2v3p87_klm5p96_basis_boosted = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=basis_boosted,
        target_point="kv2p12_k2v3p87_klm5p96",
        morphing_type="exact",
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kvm1p6_k2v2p72_klm1p36_all_points = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p6_k2v2p72_klm1p36",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kvm1p21_k2v1p94_klm0p94_all_points = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kvm1p21_k2v1p94_klm0p94",
        morphing_type="fit",
    )

    analysis_inst.x.hist_hooks.hh_vbf_fit_morphing_kv2p12_k2v3p87_klm5p96_all_points = partial(
        general_higgs_morphing,
        production_channel="vbf",
        guidance_points=[
            "kv1_k2v1_kl1", "kv1_k2v0_kl1",
            "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
            "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
            "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
            "kvm1p83_k2v3p57_klm3p39", "kv2p12_k2v3p87_klm5p96",
        ],
        target_point="kv2p12_k2v3p87_klm5p96",
        morphing_type="fit",
    )

    ###############################################################################
    ###############################################################################
    ###############################################################################

    import itertools

    for i, combination in enumerate(itertools.combinations([
        "kv1_k2v1_kl1", "kv1_k2v0_kl1",
        "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
        "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
        "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
        "kvm1p83_k2v3p57_klm3p39",
    ], 6)):
        analysis_inst.x.hist_hooks[f"hh_vbf_exact_morphing_kv2p12_k2v3p87_klm5p96_basis_{i}"] = partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=list(combination),
            target_point="kv2p12_k2v3p87_klm5p96",
            morphing_type="exact",
        )

# Remark on new Process for morphing: single ID for all morphed processes, but different names
# in order to be able to distinguish them in the plot functions, maybe need to find a way to create
# a unique ID for each morphed process.


