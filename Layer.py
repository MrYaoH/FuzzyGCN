import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FuzzyGraphConv(nn.Module):
    def __init__(self, in_features, out_features, mm_shape='triangle',
                 bias=True, int_prod=False, dfz=False, alpha=None):
        super(FuzzyGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mm_shape = mm_shape
        self.bias = bias
        self.int_prod = int_prod
        self.dfz = dfz
        self.alpha = alpha

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        if self.mm_shape == 'triangle':
            self.w_b = nn.Parameter(torch.FloatTensor(size=(self.in_features, self.out_features)), requires_grad=True)
            self.w_a = nn.Parameter(torch.FloatTensor(size=(self.in_features, self.out_features)), requires_grad=True)
            nn.init.uniform_(self.w_a, 0.02, 0.05)
            self.w_c = nn.Parameter(torch.FloatTensor(size=(self.in_features, self.out_features)), requires_grad=True)
            nn.init.uniform_(self.w_c, 0.02, 0.05)

            if self.bias is True:
                self.b_b = nn.Parameter(torch.FloatTensor(size=(1, self.out_features)), requires_grad=True)
                self.b_a = nn.Parameter(torch.FloatTensor(size=(1, self.out_features)), requires_grad=True)
                nn.init.uniform_(self.b_a, 0.02, 0.05)
                self.b_c = nn.Parameter(torch.FloatTensor(size=(1, self.out_features)), requires_grad=True)
                nn.init.uniform_(self.b_c, 0.02, 0.05)
        else:
            self.w_m = nn.Parameter(torch.FloatTensor(size=(self.in_features, self.n_rules)), requires_grad=True)
            self.w_std = nn.Parameter(torch.FloatTensor(size=self.Cs.size()), requires_grad=True)

        self.reset_parameters()

    def forward(self, input, adj):
        if self.mm_shape == 'triangle':
            if self.alpha is None:
                output = self.calculate(input, adj)
            else:
                output = self.calculate_alpha(input, adj)
        return output

    def limit_parameters(self, param, lower=0., upper=1.):
        a, c = param
        a = F.relu(a)
        c = F.relu(c)
        return a, c

    def gcn(self, input, adj, weight, bias):
        hidden = torch.mm(input, weight)
        output = torch.spmm(adj, hidden)
        if bias is not None:
            output = output + bias
        return output

    def fuzzy_scope(self, input, scope):
        if self.alpha is None:
            return torch.mm(torch.abs(input), scope)
        else:
            return torch.einsum('ij,jkl->ikl', torch.abs(input), scope)

    def fuzzyify(self, input, adj, weight_bias_alpha=None):
        center = self.gcn(input, adj, self.w_b, self.b_b)
        if self.alpha is None:
            scope_l = self.fuzzy_scope(input, self.w_a)
            scope_r = self.fuzzy_scope(input, self.w_c)
            scope_l += self.b_a
            scope_r += self.b_c

            HL = center - scope_l
            HR = center + scope_r
        else:
            k = self.alpha.shape[0]

            w_alpha, b_alpha = weight_bias_alpha
            w_b_alpha, w_a_alpha, w_c_alpha = w_alpha
            b_b_alpha, b_a_alpha, b_c_alpha = b_alpha

            scope_l = self.fuzzy_scope(input, w_a_alpha)
            scope_r = self.fuzzy_scope(input, w_c_alpha)
            scope_l += b_a_alpha
            scope_r += b_c_alpha

            center = center.unsqueeze(dim=2).expand(center.shape[0], center.shape[1], k)
            HL = center - scope_l
            HR = center + scope_r

        return center, HL, HR

    def defuzzify(self, center, left, right):
        if self.alpha is None:
            return (1 * center + left + right) / 3.
        else:
            one_minus_alpha = 1. - self.alpha

            H_alpha = (left + right) * self.alpha
            sum_alpha = 2. * torch.sum(self.alpha)

            return torch.sum(H_alpha, dim=2) / sum_alpha

    def interval_prod(self, interval_matrix1, interval_matrix2, bias=None):

        m1_l, m1_r = interval_matrix1
        m2_l, m2_r = interval_matrix2
        row = m1_l.shape[0]
        col = m2_l.shape[1]

        if self.alpha is None:
            m1_l_train = m1_l
            m1_r_train = m1_r

            m1_l_reshape = torch.reshape(m1_l_train, (1, -1))
            m1_r_reshape = torch.reshape(m1_r_train, (1, -1))

            m2_l_reshape = torch.reshape(m2_l.unsqueeze(dim=0).expand(row, m2_l.shape[0], m2_l.shape[1]),
                                         (-1, col))
            m2_r_reshape = torch.reshape(m2_r.unsqueeze(dim=0).expand(row, m2_r.shape[0], m2_r.shape[1]),
                                         (-1, col))

            mt_l = m2_l_reshape.transpose(0, 1)
            mt_r = m2_r_reshape.transpose(0, 1)

            temp1 = m1_l_reshape * mt_l
            temp2 = m1_l_reshape * mt_r
            temp3 = m1_r_reshape * mt_l
            temp4 = m1_r_reshape * mt_r
            temp = torch.stack((temp1, temp2, temp3, temp4), dim=0)
            temp_min = torch.min(temp, dim=0)[0]
            temp_max = torch.max(temp, dim=0)[0]
            temp_min_t = temp_min.transpose(0, 1)
            temp_max_t = temp_max.transpose(0, 1)
            temp_min_reshape = torch.reshape(temp_min_t, (row, m2_l.shape[0], m2_l.shape[1]))
            temp_max_reshape = torch.reshape(temp_max_t, (row, m2_l.shape[0], m2_l.shape[1]))
            output_l = torch.sum(temp_min_reshape, dim=1)
            output_r = torch.sum(temp_max_reshape, dim=1)
            if bias is not None:
                b_l, b_r = bias
                output_l += b_l
                output_r += b_r
        else:
            m1_l_train = m1_l
            m1_r_train = m1_r

            m1_l_reshape = torch.reshape(m1_l_train, (-1, m1_l.shape[-1]))
            m1_r_reshape = torch.reshape(m1_r_train, (-1, m1_r.shape[-1]))

            m2_l_reshape = \
                torch.reshape(m2_l.unsqueeze(dim=0).expand(row, m2_l.shape[0], m2_l.shape[1], m2_l.shape[2]),
                              (-1, col, m2_l.shape[2]))
            m2_r_reshape = \
                torch.reshape(m2_r.unsqueeze(dim=0).expand(row, m2_r.shape[0], m2_r.shape[1], m2_r.shape[2]),
                              (-1, col, m2_l.shape[2]))

            mt_l = m2_l_reshape.transpose(0, 1)
            mt_r = m2_r_reshape.transpose(0, 1)

            temp1 = m1_l_reshape * mt_l
            temp2 = m1_l_reshape * mt_r
            temp3 = m1_r_reshape * mt_l
            temp4 = m1_r_reshape * mt_r

            temp = torch.stack((temp1, temp2, temp3, temp4), dim=0)
            temp_min = torch.min(temp, dim=0)[0]
            temp_max = torch.max(temp, dim=0)[0]
            temp_min_t = temp_min.transpose(0, 1)
            temp_max_t = temp_max.transpose(0, 1)

            temp_min_reshape = torch.reshape(temp_min_t, (row, m2_l.shape[0], m2_l.shape[1], m2_l.shape[2]))
            temp_max_reshape = torch.reshape(temp_max_t, (row, m2_l.shape[0], m2_l.shape[1], m2_l.shape[2]))
            output_l = torch.sum(temp_min_reshape, dim=1)
            output_r = torch.sum(temp_max_reshape, dim=1)
            if bias is not None:
                b_l, b_r = bias
                output_l += b_l
                output_r += b_r
        return output_l, output_r

    def adj_prod_feature(self, hl, hr, adj):
        if self.alpha is None:
            HL = torch.spmm(adj, hl)
            HR = torch.spmm(adj, hr)
        else:
            HL = torch.reshape(torch.spmm(adj, torch.reshape(hl, (hl.shape[0], -1))),
                               (hl.shape[0], hl.shape[1], hl.shape[2]))
            HR = torch.reshape(torch.spmm(adj, torch.reshape(hr, (hr.shape[0], -1))),
                               (hr.shape[0], hr.shape[1], hr.shape[2]))
        return HL, HR

    def weight_loss(self):
        param = [self.w_b, self.b_b]
        w_loss = 0.
        for p in param:
            w_loss += torch.sum(p * p)
        return 0.5 * w_loss

    def fuzziness_loss(self):
        param = [self.w_a, self.w_c, self.b_a, self.b_c]
        w_loss = 0.
        for p in param:
            w_loss += torch.sum(p * p)
        return 0.5 * w_loss

    def reset_parameters(self):
        stdv = 1 / np.sqrt(self.w_b.size(1))
        self.w_b.data.uniform_(-stdv, stdv)
        if self.b_b is not None:
            self.b_b.data.uniform_(-stdv, stdv)

    def calculate(self, input, adj):
        w_a, w_c = self.limit_parameters((self.w_a, self.w_c))
        b_a, b_c = self.limit_parameters((self.b_a, self.b_c))

        if self.int_prod is False:  # fuzzy graph convolution process
            center, HL, HR = self.fuzzyify(input, adj)
        else:
            w_l = self.w_b - w_a
            w_r = self.w_b + w_c
            input_w = (w_l, w_r)
            b_l = self.b_b - b_a
            b_r = self.b_b + b_c
            input_b = (b_l, b_r)
            center, hl, hr = input
            H = self.adj_prod_feature(hl, hr, adj)
            output_l, output_r = self.interval_prod(H, input_w, input_b)

        if self.dfz is True:
            if self.int_prod is False:
                output = (center, HL, HR)
            else:
                output = (output_l, output_r)
        else:
            output = center, HL, HR
        return output

    def calculate_alpha(self, input, adj):

        k = self.alpha.shape[0]

        w_a, w_c = self.limit_parameters((self.w_a, self.w_c))
        b_a, b_c = self.limit_parameters((self.b_a, self.b_c))

        one_minus_alpha = 1. - self.alpha
        w_b_alpha = self.w_b.unsqueeze(dim=2).expand(self.w_b.shape[0], self.w_b.shape[1], k)
        w_a_alpha = w_a.unsqueeze(dim=2).expand(w_a.shape[0], w_a.shape[1], k)
        w_c_alpha = w_c.unsqueeze(dim=2).expand(w_c.shape[0], w_c.shape[1], k)

        b_b_alpha = self.b_b.expand(k, self.b_b.shape[1]).transpose(0, 1)
        b_a_alpha = b_a.expand(k, b_a.shape[1]).transpose(0, 1)
        b_c_alpha = b_c.expand(k, b_c.shape[1]).transpose(0, 1)

        for i in range(1, k):
            w_a_alpha[:, :, i] = w_a_alpha[:, :, 0] * one_minus_alpha[i]
            w_c_alpha[:, :, i] = w_c_alpha[:, :, 0] * one_minus_alpha[i]
            b_a_alpha[:, i] = b_a_alpha[:, i] * one_minus_alpha[i]
            b_c_alpha[:, i] = b_c_alpha[:, i] * one_minus_alpha[i]

        w_a_alpha = torch.reshape(w_a_alpha, (w_a_alpha.shape[0], w_a_alpha.shape[1], -1))
        w_c_alpha = torch.reshape(w_c_alpha, (w_c_alpha.shape[0], w_c_alpha.shape[1], -1))
        b_a_alpha = torch.reshape(b_a_alpha, (b_a_alpha.shape[0], -1))
        b_c_alpha = torch.reshape(b_c_alpha, (b_c_alpha.shape[0], -1))

        weight_alpha = (w_b_alpha, w_a_alpha, w_c_alpha)
        bias_alpha = (b_b_alpha, b_a_alpha, b_c_alpha)
        weight_bias_alpha = (weight_alpha, bias_alpha)

        if self.int_prod is False:
            center, HL, HR = self.fuzzyify(input, adj, weight_bias_alpha)
        else:
            w_l_alpha = w_b_alpha - w_a_alpha
            w_r_alpha = w_b_alpha + w_c_alpha
            input_w = (w_l_alpha, w_r_alpha)

            b_l_alpha = b_b_alpha - b_a_alpha
            b_r_alpha = b_b_alpha + b_c_alpha
            input_b = (b_l_alpha, b_r_alpha)

            center, hl, hr = input
            H = self.adj_prod_feature(hl, hr, adj)
            output_l, output_r = self.interval_prod(H, input_w, input_b)


        if self.dfz is True:
            if self.int_prod is False:
                output = (center, HL, HR)
            else:
                output = (output_l, output_r)
        else:
            output = center, HL, HR
        return output

    def __repr__(self):
        return self.__class__.__name__ + "{} -> {}".format(self.in_features, self.out_features)


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(size=(in_features, out_features)), requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(1, out_features)), requires_grad=True)

        self.in_features = in_features
        self.out_features = out_features

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hidden = torch.mm(input, self.weight)
        output = torch.spmm(adj, hidden)
        if self.bias is not None:
            output = output + self.bias
        return output

    def weight_loss(self):
        param = [self.weight, self.bias]
        w_loss = 0.
        for p in param:
            w_loss += torch.sum(p * p)
        return 0.5 * w_loss

    def __repr__(self):
        return self.__class__.__name__ + "{} -> {}".format(self.in_features, self.out_features)
