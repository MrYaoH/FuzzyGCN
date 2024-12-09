from torch_geometric.nn import GCNConv
from Layer import *


class GCN(nn.Module):
    def __init__(self, input, hidden1, hidden2, dropout=0.5):
        super(GCN, self).__init__()
        self.input = input
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        self.gcn1 = GraphConv(input, hidden1)
        self.gcn2 = GraphConv(hidden1, hidden2)

    def forward(self, x, edge_index, out_loss=False):
        x = self.dropout(x)
        x = self.gcn1(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index)
        if out_loss is True:
            loss = [self.gcn1.weight_loss() + self.gcn2.weight_loss()]
            return x, loss
        return x


class FGCN(nn.Module):
    def __init__(self, input, hidden, output, x, mm_shape='triangle', fz_wd=True):
        super(FGCN, self).__init__()

        self.input = input
        self.hidden = hidden
        self.output = output
        self.fz_wd = fz_wd

        self.mm_shape = mm_shape

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda:1

        self.fgcn1 = FuzzyGraphConv(self.input, self.hidden)
        self.fgcn2 = FuzzyGraphConv(self.hidden, self.output, int_prod=True, dfz=True)

        self.act = nn.ReLU()

        self.center = nn.Parameter(nn.init.constant_(torch.FloatTensor(size=(x.shape[0], self.output)), 0.5), requires_grad=True)


    def forward(self, x, adj, out_loss=False):
        Center, HL, HR = self.fgcn1(x, adj)
        HL = self.act(HL)
        HR = self.act(HR)
        HL, HR = self.fgcn2((Center, HL, HR), adj)
        center = self.update_center(HL, HR)
        Output = center

        if out_loss is True:
            loss = [self.fgcn1.weight_loss() + self.fgcn2.weight_loss()]
            if self.fz_wd is True:
                loss.append(self.fgcn1.fuzziness_loss() + self.fgcn2.fuzziness_loss())

            return Output, loss
        return Output

    def update_center(self, hl, hr):

        center = torch.clamp(self.center, min=0., max=1.)
        if len(hl.shape) <= 2:
            return hl + center * (hr - hl)
        else:
            return hl + torch.einsum('ij,ijk->ijk', center, hr-hl)
