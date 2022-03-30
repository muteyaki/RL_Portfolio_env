import  torch
from    torch import nn
from    torch.nn import functional as F
from    layer import GraphConvolution

class GCN(nn.Module):


    def __init__(self, input_dim, hidden,output_dim):
        super(GCN, self).__init__()

        self.input_dim = input_dim # 1433
        self.output_dim = output_dim

        self.layers = nn.Sequential(GraphConvolution(self.input_dim, hidden,
                                                     activation=F.relu,
                                                     dropout=0.5,
                                                     is_sparse_inputs=True),

                                    GraphConvolution(hidden, output_dim,
                                                     activation=F.relu,
                                                     dropout=0.5,
                                                     is_sparse_inputs=False),)

    def forward(self, inputs):
        x, support = inputs

        x = self.layers((x, support))

        return x

    def l2_loss(self):

        layer = self.layers.children()
        layer = next(iter(layer))

        loss = None

        for p in layer.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()

        return loss