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

class TabNeutralAD(nn.Module):
    def __init__(self, model, x_dim,config):
        super(TabNeutralAD, self).__init__()

        self.enc,self.trans = model._make_nets(x_dim,config)
        self.num_trans = config['num_trans']
        self.trans_type = config['trans_type']
        self.device = config['device']
        try:
            self.z_dim = config['latent_dim']
        except:
            if 32<=x_dim <= 300:
                self.z_dim = 32
            elif x_dim<32:
                self.z_dim = 2 * x_dim
            else:
                self.z_dim = 64

    def forward(self,x):
        x = x.type(torch.FloatTensor).to(self.device)
        x_T = torch.empty(x.shape[0],self.num_trans,x.shape[-1]).to(x)
        for i in range(self.num_trans):
            mask = self.trans[i](x)
            if self.trans_type == 'forward':
                x_T[:, i] = mask
            elif self.trans_type == 'mul':
                mask = torch.sigmoid(mask)
                x_T[:, i] = mask * x
            elif self.trans_type == 'residual':
                x_T[:, i] = mask + x
        x_cat = torch.cat([x.unsqueeze(1),x_T],1)
        zs = self.enc(x_cat.reshape(-1,x.shape[-1]))
        zs = zs.reshape(x.shape[0],self.num_trans+1,self.z_dim)

        return zs

class SeqNeutralAD(nn.Module):
    def __init__(self, model, x_dim,config):
        super(SeqNeutralAD, self).__init__()

        self.enc,self.trans = model._make_nets(x_dim,config)
        self.num_trans = config['num_trans']
        self.trans_type = config['trans_type']
        self.device = config['device']
        self.z_dim = config['latent_dim']

    def forward(self,x):
        x = x.type(torch.FloatTensor).to(self.device)

        x_T = torch.empty(x.shape[0],self.num_trans,x.shape[1],x.shape[2]).to(x)
        for i in range(self.num_trans):
            mask = self.trans[i](x)

            if self.trans_type == 'forward':
                x_T[:, i] = mask
            elif self.trans_type == 'mul':
                mask = torch.sigmoid(mask)
                x_T[:, i] = mask * x
            elif self.trans_type == 'residual':
                x_T[:, i] = mask + x
        x_cat = torch.cat([x.unsqueeze(1),x_T],1)
        zs = self.enc[0](x_cat.reshape(-1,x.shape[1],x.shape[2]))
        zs = zs.reshape(x.shape[0],self.num_trans+1,self.z_dim)

        return zs

class FeatNeutralAD(nn.Module):
    def __init__(self, model, x_dim,config,seed):
        super(FeatNeutralAD, self).__init__()

        self.enc,self.trans = model._make_nets(x_dim,config,seed)
        self.num_trans = config['num_trans']
        self.neg_num = config['neg_num']
        self.trans_type = config['trans_type']
        self.device = config['device']
        self.z_dim = config['enc_zdim']

    def forward(self,x):
        x = x.type(torch.FloatTensor).to(self.device)
        num_trans = self.num_trans+self.neg_num
        x_T = torch.empty(x.shape[0],num_trans,x.shape[-1]).to(x)
        for i in range(num_trans):
            mask = self.trans[i](x)
            if self.trans_type == 'forward':
                x_T[:, i] = mask
            elif self.trans_type == 'mul':
                mask = torch.sigmoid(mask)
                x_T[:, i] = mask * x
            elif self.trans_type == 'residual':
                x_T[:, i] = mask + x
        x_cat = torch.cat([x.unsqueeze(1),x_T],1) # n,k+1,z
        zs,z_shift = self.enc(x_cat.reshape(-1,x.shape[-1]))
        zs = zs.reshape(x.shape[0],num_trans+1,self.z_dim)
        # z_shift = z_shift.reshape(x.shape[0]*(num_trans+1),2)
   
        # z_shift = torch.cat([z_shift[:,0].unsqueeze(1),z_shift[:,self.num_trans+1:self.neg_num]],1)
        return zs,z_shift

class VisNeutralAD(nn.Module):
    def __init__(self, model, x_dim,config,seed):
        super(VisNeutralAD, self).__init__()
        self.enc,self.trans = model._make_nets(x_dim,config)
        self.num_trans = config['num_trans']
        self.neg_num = config['neg_num']
        self.trans_type = config['trans_type']
        self.device = config['device']
        self.z_dim =  config['enc_zdim']

    def forward(self, x):
        x = x.type(torch.FloatTensor).to(self.device)
        num_trans = self.num_trans + self.neg_num
        x_T = torch.empty(x.shape[0],num_trans,x.shape[1],x.shape[2],x.shape[3]).to(x)
        for i in range(num_trans):
            mask = self.trans[i](x)
            if self.trans_type == 'forward':
                mask = torch.tanh(mask)
                x_T[:, i] = mask
            elif self.trans_type == 'mul':
                mask = torch.sigmoid(mask)
                x_T[:, i] = mask * x
            elif self.trans_type == 'residual':
                x_T[:, i] = mask + x
        x_cat = torch.cat([x.unsqueeze(1),x_T],1)
        zs, z_shift = self.enc(x_cat.reshape(-1,x.shape[1],x.shape[2],x.shape[3]))
        zs = zs.reshape(x.shape[0],num_trans+1,self.z_dim)

        return zs,z_shift

