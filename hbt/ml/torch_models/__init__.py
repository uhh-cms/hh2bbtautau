from hbt.ml.torch_models.binary import (
    FeedForwardNet,
    TensorFeedForwardNet,
    TensorFeedForwardNetAdam,
    FeedForwardArrow,
    DropoutFeedForwardNet,
    WeightedTensorFeedForwardNet,
    WeightedTensorFeedForwardNetWithCat,
    WeightedTensorFeedForwardNetWithCatReducedEmbedding,
    WeightedTensorFeedForwardNetWithCatReducedEmbedding1F,
    WeightedTensorFeedForwardNetWithCatOutsourceTokens,
)
from hbt.ml.torch_models.multi_class import (
    FeedForwardMultiCls,
    WeightedFeedForwardMultiCls,
    DeepFeedForwardMultiCls,
)
from hbt.ml.torch_models.resnet import (
    ResNet,
    WeightedResNet,
    WeightedResnetNoDropout,
    WeightedResnetTest,
    WeightedResnetTest2,
)
from hbt.ml.torch_models.bognet import (
    BogNet,
)

model_clss = {}
model_clss["feedforward"] = FeedForwardNet
model_clss["feedforward_tensor"] = TensorFeedForwardNet
model_clss["feedforward_tensor_adam"] = TensorFeedForwardNetAdam
model_clss["feedforward_arrow"] = FeedForwardArrow
model_clss["weighted_feedforward_tensor"] = WeightedTensorFeedForwardNet
model_clss["weighted_feedforward_tensor_cat"] = WeightedTensorFeedForwardNetWithCat
model_clss["weighted_feedforward_tensor_cat_outsource"] = WeightedTensorFeedForwardNetWithCatOutsourceTokens
model_clss["weighted_feedforward_tensor_cat_reduced_embedding"] = WeightedTensorFeedForwardNetWithCatReducedEmbedding
model_clss["weighted_feedforward_tensor_cat_reduced_embedding1f"] = WeightedTensorFeedForwardNetWithCatReducedEmbedding1F


# multi class networks
model_clss["feedforward_multicls"] = FeedForwardMultiCls
model_clss["weighted_feedforward_multicls"] = WeightedFeedForwardMultiCls
model_clss["deepfeedforward_multicls"] = DeepFeedForwardMultiCls
model_clss["feedforward_dropout"] = DropoutFeedForwardNet

# resnet networks
model_clss["resnet"] = ResNet
model_clss["weighted_resnet"] = WeightedResNet
model_clss["weighted_resnet_nodroupout"] = WeightedResnetNoDropout
model_clss["weighted_resnet_test"] = WeightedResnetTest
model_clss["weighted_resnet_test2"] = WeightedResnetTest2
model_clss["bognet"] = BogNet
