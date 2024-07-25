import order as od
from columnflow.util import maybe_import
from columnflow.columnar_util import fill_hist

hist = maybe_import("hist")


def create_regions(rate, rate_int_dic, rate_int, cat_id, regions, mc_hist, data_hist, cat_name, cat_parent):
    if "os" in cat_name:
        if "noniso" not in cat_name:
            regions[cat_parent]["A"] = {}
            regions[cat_parent]["A"]["mc"] = mc_hist[{"category": hist.loc(cat_id)}]
            regions[cat_parent]["A"]["data"] = data_hist[{"category": hist.loc(cat_id)}]
            # regions[cat_parent]["A"]["mc"] = mc_hist.counts()[index]
            # regions[cat_parent]["A"]["data"] = data_hist.counts()[index]
            print("... created region A for category " + cat_parent)
        else:
            regions[cat_parent]["B"] = {}
            regions[cat_parent]["B"]["mc"] = mc_hist[{"category": hist.loc(cat_id)}]
            regions[cat_parent]["B"]["data"] = data_hist[{"category": hist.loc(cat_id)}]

            rate_int_dic[cat_parent]["B"] = {}
            rate_int_dic[cat_parent]["B"] = rate_int[{"category": hist.loc(cat_id)}].values()[0]
            rate_int_dic[cat_parent]["Bshape"] = rate[{"category": hist.loc(cat_id)}]
            print("... created region B for category " + cat_parent)
    elif "ss" in cat_name:
        if "noniso" not in cat_name:
            regions[cat_parent]["C"] = {}
            regions[cat_parent]["C"]["mc"] = mc_hist[{"category": hist.loc(cat_id)}]
            regions[cat_parent]["C"]["data"] = data_hist[{"category": hist.loc(cat_id)}]

            rate_int_dic[cat_parent]["C"] = {}
            rate_int_dic[cat_parent]["C"] = rate_int[{"category": hist.loc(cat_id)}].values()[0]
            print("... created region C for category " + cat_parent)
        else:
            regions[cat_parent]["D"] = {}
            regions[cat_parent]["D"]["mc"] = mc_hist[{"category": hist.loc(cat_id)}]
            regions[cat_parent]["D"]["data"] = data_hist[{"category": hist.loc(cat_id)}]

            rate_int_dic[cat_parent]["D"] = {}
            rate_int_dic[cat_parent]["D"] = rate_int[{"category": hist.loc(cat_id)}].values()[0]
            print("... created region D for category " + cat_parent)


def add_hooks(cfg: od.Config):

    def dev_ana_ABCD(task, hists):

        mc_hists = [h for p, h in hists.items() if p.is_mc and not p.name.startswith("hh")]
        data_hists = [h for p, h in hists.items() if p.is_data]

        if mc_hists and data_hists:

            mc_hist = sum(mc_hists[1:], mc_hists[0].copy())
            data_hist = sum(data_hists[1:], data_hists[0].copy())

            # calculate gap between data and mc in all ABCD regions
            rate = data_hist + ((-1) * mc_hist)
            rate_int = rate.integrate(rate.axes[2].name)

            # check if the number of categories is multiple of 4
            cat_count = mc_hist.axes[0].size
            if cat_count % 4 != 0:
                raise ValueError("Number of categories is not a multiple of 4")

            # initialise regions and rates dictionaries
            regions = {}
            rate_int_dic = {}

            for index in range(cat_count):
                # get corresponding category id, name and parent category
                cat_id = mc_hist.axes[0].value(index)
                cat_name = cfg.get_category(cat_id).name
                cat_parent = cat_name.partition("__")[0]
                print("... processing category " + str(cat_id) + " " + cat_name + " (cat_parent is " + cat_parent + ")")

                # define ABCD regions
                if cat_parent not in regions:
                    regions[cat_parent] = {}
                    rate_int_dic[cat_parent] = {}
                    create_regions(rate, rate_int_dic, rate_int, cat_id, regions, mc_hist, data_hist, cat_name, cat_parent)
                else:
                    create_regions(rate, rate_int_dic, rate_int, cat_id, regions, mc_hist, data_hist, cat_name, cat_parent)

            # NOTE: the ABCD regions have the same entries for both parent categories (incl and 2j) why ?

            # initialise empty QCD histograms
            qcd_hist = rate.copy()
            qcd_hist = qcd_hist.reset()

            scale_factor = {}
            for cat_parent in regions.keys():
                scale_factor[cat_parent] = rate_int_dic[cat_parent]["C"] / rate_int_dic[cat_parent]["D"]
                print("... scale factor for " + cat_parent + " parent category is " + str(scale_factor[cat_parent]))

            from IPython import embed
            embed()
            quit()

            # add QCD histogram to hists
            # for cat_parent="incl":
            # Bshape = rate_int_dic[cat_parent]["Bshape"]
            # TODO: BUG in fill_hist function (ValueError: got multi-dimensional hist but only one dimensional data)
            # hists[cfg.processes.n.qcd] = fill_hist(qcd_hist, Bshape) * scale_factor[cat_parent]

            return hists

    cfg.x.hist_hooks = {
        "dev_ana_ABCD": dev_ana_ABCD,
    }
