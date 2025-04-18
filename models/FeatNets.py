# Neural Transformation Learning for Anomaly Detection (NeuTraLAD) - a self-supervised method for anomaly detection
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import torch
import torch.nn as nn

class FeatTransformNet(nn.Module):
    def __init__(self, x_dim,h_dim,num_layers):
        super(FeatTransformNet, self).__init__()
        net = []
        input_dim = x_dim
        for _ in range(num_layers-1):
            net.append(nn.Linear(input_dim,h_dim,bias=False))
            net.append(nn.BatchNorm1d(h_dim,affine=False))
            net.append(nn.ReLU())
            input_dim= h_dim
        net.append(nn.Linear(input_dim,x_dim,bias=False))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        out = self.net(x)

        return out


class FeatEncoder(nn.Module):
    def __init__(self, x_dim,z_dim,bias,num_layers,batch_norm):

        super(FeatEncoder, self).__init__()

        enc = []
        input_dim = x_dim
        for _ in range(num_layers - 1):
            enc.append(nn.Linear(input_dim, int(input_dim/2),bias=bias))
            if batch_norm:
                enc.append(nn.BatchNorm1d(int(input_dim/2),affine=bias))
            enc.append(nn.ReLU())
            input_dim = int(input_dim/2)

        self.fc = nn.Linear(input_dim, z_dim,bias=bias)
        self.shift_cls_layer = nn.Linear(input_dim, 2,bias=bias)
        self.enc = nn.Sequential(*enc)



    def forward(self, x):

        z = self.enc(x)
        z_shift = self.shift_cls_layer(z)
        z = self.fc(z)

        return z,z_shift



class FeatNets():

    def _make_nets(self,x_dim,config,seed=100):
        enc_nlayers = config['enc_nlayers']
        z_dim = config['enc_zdim']

        trans_nlayers = config['trans_nlayers']
        trans_hdim = config['trans_hdim']
        neg_num = config['neg_num']
        num_trans = config['num_trans']
        batch_norm = config['batch_norm']
        # c_seed = torch.seed()
        # print(f"{c_seed = }")
        # torch.manual_seed(c_seed)
        # torch.manual_seed(42)
        enc = FeatEncoder(x_dim,  z_dim, config['enc_bias'],enc_nlayers,batch_norm)
        # torch.manual_seed(seed)
        trans = nn.ModuleList(
            [FeatTransformNet(x_dim, trans_hdim, trans_nlayers) for _ in range(num_trans+neg_num)])
        return enc,trans

