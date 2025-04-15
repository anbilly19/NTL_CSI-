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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def SLoss():
    return nn.CrossEntropyLoss(reduction='none')

class NegDCL(nn.Module):
    def __init__(self, temperature=0.1):
        super(NegDCL, self).__init__()
        self.temp = temperature
        
    def forward(self, z, eval=False):
        z = F.normalize(z, p=2, dim=-1)
        z_ori = z[:, 0]  # n,z
        z_trans = z[:, 1:]  # n,k-1, z
        
        batch_size, num_trans, z_dim = z.shape
        
        # Determine how many transformations to use as positive samples
        num_pos_trans = (num_trans - 1) // 2
        num_neg_trans = num_trans - 1 - num_pos_trans
        
        # Split transformations into positive and negative samples
        z_trans_pos = z_trans[:, :num_pos_trans]  # n,pos,z
        z_trans_neg = z_trans[:, num_pos_trans:]  # n,neg,z
        
        # Compute similarity matrix
        sim_matrix = torch.exp(torch.matmul(z, z.permute(0, 2, 1)) / self.temp)  # n,k,k
        mask = (torch.ones_like(sim_matrix).to(z) - torch.eye(num_trans).unsqueeze(0).to(z)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size, num_trans, -1)
        
        # Compute similarities for positive transformations only
        pos_sim = torch.exp(torch.sum(z_trans_pos * z_ori.unsqueeze(1), -1) / self.temp)  # n,pos
        
        # Get the transformation matrix for positive transformations
        trans_matrix_pos = sim_matrix[:, 1:num_pos_trans+1].sum(-1)  # n,pos
        
        # For negative transformations, we want to minimize similarity
        neg_sim = torch.exp(torch.sum(z_trans_neg * z_ori.unsqueeze(1), -1) / self.temp)  # n,neg
        trans_matrix_neg = sim_matrix[:, num_neg_trans+1:].sum(-1)  # n,neg
        
        # Positive loss (minimize negative log likelihood)
        K_pos = num_pos_trans
        scale_pos = 1 / np.abs(K_pos * np.log(1.0 / K_pos)) #if K_pos > 1 else 0
        pos_loss = (torch.log(trans_matrix_pos) - torch.log(pos_sim)) * scale_pos
        
        # Negative loss (maximize negative log likelihood)
        K_neg = num_neg_trans
        scale_neg = 1 / np.abs(K_neg * np.log(1.0 / K_neg)) #if K_neg > 1 else 0
        neg_loss = 1 * (torch.log(neg_sim) + torch.log(trans_matrix_neg)) * scale_neg  # Minimize similarity for negative samples
        
        # Combine losses
        if num_pos_trans > 0:
            pos_loss = pos_loss.sum(1)
        else:
            pos_loss = torch.zeros(batch_size).to(z)
            
        if num_neg_trans > 0:
            neg_loss = neg_loss.sum(1)
        else:
            neg_loss = torch.zeros(batch_size).to(z)
        
        # Total loss
        dcl = pos_loss + neg_loss
        
        return dcl

class DCL(nn.Module):
    def __init__(self,temperature=0.1):
        super(DCL, self).__init__()
        self.temp = temperature
    def forward(self,z,eval=False):
        z = F.normalize(z, p=2, dim=-1)
        z_ori = z[:, 0]  # n,z
        z_trans = z[:, 1:]  # n,k-1, z
        batch_size, num_trans, z_dim = z.shape

        sim_matrix = torch.exp(torch.matmul(z, z.permute(0, 2, 1) / self.temp))  # n,k,k
        mask = (torch.ones_like(sim_matrix).to(z) - torch.eye(num_trans).unsqueeze(0).to(z)).bool() # diagonal removal
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size, num_trans, -1)
        trans_matrix = sim_matrix[:, 1:].sum(-1)  # n,k-1

        pos_sim = torch.exp(torch.sum(z_trans * z_ori.unsqueeze(1), -1) / self.temp) # n,k-1
        K = num_trans - 1
        scale = 1 / np.abs(K*np.log(1.0 / K))

        loss_tensor = (torch.log(trans_matrix) - torch.log(pos_sim)) * scale

        dcl = loss_tensor.sum(1)
        return dcl

class EnhancedDCL(nn.Module):
    def __init__(self, temperature=0.1, lambda_cross=1.0):
        super(EnhancedDCL, self).__init__()
        self.temp = temperature
        self.lambda_cross = lambda_cross  # Weight for cross-sample negative term
        
    def forward(self, z, eval=False):
        z = F.normalize(z, p=2, dim=-1)
        z_ori = z[:, 0]  # n,z
        z_trans = z[:, 1:]  # n,k-1,z
        batch_size, num_trans, z_dim = z.shape
        
        # Within-sample similarities (original DCL part)
        sim_matrix = torch.exp(torch.matmul(z, z.permute(0, 2, 1)) / self.temp)  # n,k,k
        mask = (torch.ones_like(sim_matrix).to(z) - torch.eye(num_trans).unsqueeze(0).to(z)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size, num_trans, -1)
        trans_matrix = sim_matrix[:, 1:].sum(-1)  # n,k-1
        
        pos_sim = torch.exp(torch.sum(z_trans * z_ori.unsqueeze(1), -1) / self.temp)  # n,k-1
        
        # Cross-sample negative term (new addition)
        # Compute similarities between each sample's original and other samples' transformations
        cross_sample_sim = torch.matmul(z_ori, z_trans.reshape(z_trans.shape[0]*(num_trans-1),z_dim).t())  # n,n*(k-1)
        cross_sample_sim = torch.exp(cross_sample_sim / self.temp)
        
        # Create mask to exclude positive pairs (same sample)
        sample_indices = torch.arange(batch_size, device=z.device)
        mask_cross = torch.ones(batch_size, batch_size * (num_trans - 1), device=z.device)
        
        # For each row i, mask out columns that belong to the same sample i
        for i in range(batch_size):
            start_idx = i * (num_trans - 1)
            end_idx = start_idx + (num_trans - 1)
            mask_cross[i, start_idx:end_idx] = 0
            
        cross_neg_sim = (cross_sample_sim * mask_cross).sum(1).unsqueeze(1).expand(-1, num_trans - 1) # (1,k-1)
        
        # Combine with original loss
        K = num_trans - 1
        scale = 1 / np.abs(K * np.log(1.0 / K))
        
        # Original loss term
        original_loss = (torch.log(trans_matrix) - torch.log(pos_sim)) * scale
        
        # New cross-sample term (maximizing difference between non-matching samples)
        cross_loss = torch.log(cross_neg_sim + trans_matrix) - torch.log(pos_sim)
        
        # Combined loss
        loss_tensor = original_loss + self.lambda_cross * cross_loss
        
        dcl = loss_tensor.sum(1)
        return dcl

class EucDCL(nn.Module):
    def __init__(self, temperature=1):
        super().__init__()
        self.temp = temperature

    def forward(self, z, eval=False):

        batch_size, num_trans, z_dim = z.shape
        sim_matrix = -torch.cdist(z,z)

        sim_matrix = torch.exp(sim_matrix / self.temp)  # n,k,k
        mask = (torch.ones_like(sim_matrix).to(z) - torch.eye(num_trans).unsqueeze(0).to(z)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size, num_trans, -1)
        sim_matrix = sim_matrix+1e-8
        trans_matrix = sim_matrix[:, 1:].sum(-1)  # n,k-1
        pos_sim = sim_matrix[:, 1:, 0]

        K = num_trans - 1
        scale = 1 / np.abs(K*np.log(1.0 / K))
        score = (-torch.log(pos_sim) + torch.log(trans_matrix)) * scale
        if eval:
            score = score.sum(1)
            return score
        else:
            loss = score.sum(1)
            return loss

