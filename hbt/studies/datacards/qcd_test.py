# coding: utf-8

import os

import hist

from columnflow.inference import InferenceModel, ParameterType, FlowStrategy
from columnflow.inference.cms.datacard import DatacardWriter


class QCDModel(InferenceModel):

    def init_func(self):
        self.add_category(
            "single_category",
            data_from_processes=["tt", "qcd"],  # make up fake data from tt + qcd
            mc_stats=10,  # bb/bb-lite threshold
            empty_bin_value=0.0,  # disables empty bin filling
            flow_strategy=FlowStrategy.warn,  # warn if under/overflow bins have non-zero content
        )

        self.add_process(name="hh", is_signal=True)
        self.add_process(name="tt", is_signal=False)
        self.add_process(name="qcd", is_signal=False)

        self.add_parameter("BR_hbb", type=ParameterType.rate_gauss, process="hh", effect=(0.9874, 1.0124))
        self.add_parameter("pdf_gg", type=ParameterType.rate_gauss, process="tt", effect=1.042)


def create_hist(values, variances, flow=False):
    assert len(values) > (2 if flow else 0)
    assert len(values) == len(variances)
    h = hist.Hist.new.Reg(len(values) - (2 if flow else 0), 0.0, 1.0, name="x").Weight()
    h.view(flow=flow).value[...] = values
    h.view(flow=flow).variance[...] = variances
    return h


# dummy histograms, 2 bins
h_hh = create_hist([0.5, 0.5], [0.1, 0.1])
h_tt = create_hist([10.0, 0.25], [1.0, 0.1])
h_qcd = create_hist([5.1, 0.0], [0.5, 0.0])

# histogram structure expected by the datacard writer
# category -> process -> config -> shift -> hist
datacard_hists = {
    "single_category": {
        "hh": {"22pre_v14": {"nominal": h_hh}},
        "tt": {"22pre_v14": {"nominal": h_tt}},
        "qcd": {"22pre_v14": {"nominal": h_qcd}},
    },
}

# write it
qcd_model = QCDModel()
writer = DatacardWriter(qcd_model, datacard_hists)
this_dir = os.path.dirname(os.path.abspath(__file__))
writer.write(
    os.path.join(this_dir, "datacard.txt"),
    os.path.join(this_dir, "shapes.root"),
    shapes_path_ref="shapes.root",
)
